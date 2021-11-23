# transformer-explainer

transformer-explainer is a simple library, which provides multiple explanation methods and HTML visualisation methods. It has been created for a specific NLP binary-classification model, but the implementation is very simple and can be easily modified to support other models like multinomial classifiers. 

The library is quite specific in its implementation (details below), but the implementation is simple enough to be *hopefully* easily modifiable.

The library was tested with TF2 and Huggingface BERT model.

## Downloading the model

The model is too large to be uploaded to github, so it is made public on https://drive.google.com/file/d/1AbFSgM_bWntFIYu_lWF1r1R8boD2pCJT/view?usp=sharing as a .zip archive, which is to be extracted to the root of the project.

## Attribution methods

The library supports multiple attribution methods - basic one like Gradients or Gradients*Input and more complex ones like Integrated Gradients or SmoothGRAD. The methods in general are sufficiently customizable. The implementations are simple, so modifying the methods to support a specific model should not be too complex.

The attribution methods rely on a Huggingface model feature, where instead of <code>input_ids</code> the embeddings can be passed as <code>input_embeds</code> parameter, bypassing the embedding operations in the model. The library uses it in all of the explanation methods so the model has to support it. The library right now does not support passing of any <code>**kwargs</code> to the model, but it should be simple to modify if need be.

In summary, a tuple of three tensors is passed to the model - <code>(input_embeds, attention_mask, token_type_ids)</code>. The model thus needs to resolve the non-standard <code>input_embeds</code> in the tuple and pass it to the transformer model correctly - this can be (I hope) in most cases solved by a simple wrapper.

Also, the attributions are produced for the tensor passed as <code>input_embeds</code>, which usually has shape <code>(sequence_length, embedding_dim)</code>, which means that for each token there are <code>embedding_dim</code> attributions, which are then summed.

And last, he attribution methods expand the inputs to have a batch dimension, which the model this library was made for required - e.g. for <code>input_embeds</code> with shape <code>(1536, 768)</code> a batch dimension is added to create shape <code>(1, 1536, 768)</code>. This is done for each input, but it is realized trough a single function in <code>utils.py</code> (<code>to_tensor_and_expand</code>), which can be modified.

All the attribution methods have to be instantiated before use. The attribution methods always require a model to be passed to the constructor. The model must have a <code>__call__</code> method. All the attribution methods have <code>explain</code> method, which produces and <code>Explanation</code> instance, that contains the tokens, their attributions and classification result. The <code>explain</code> method always has these parameters:
<ul>
	<li><code>input_embeds</code> - Input embeddings of the passed input.</li>
	<li><code>attention_mask</code></li>
	<li><code>token_type_ids</code></li>
	<li><code>tokens</code> - Tokenized input (<code>list</code> or <code>array-like</code>), so that the <code>Explanation</code> instance can be produced.</li>
	<li><code>silent</code> - Whether or not to print status messages into stdout. This can be useful when doing batch processing with JSONized <code>Explanation</code> being printed into stdout redirected to file.</li>
</ul>

All the methods are implemented in <code>transformer_explainer/explainers</code>.

### Gradients and Gradients*Input
The pure gradients and gradients*input methods are implemented in <code>gradients.py</code> as <code>ExplainerGradients</code> class and simply take the gradients of the network output w.r.t. input tensor (supplied as <code>input_embeds</code>).

Whether to use gradients or gradients*input is specified trough the <code>x_inputs</code> parameter (default is <code>False</code>). If set to <code>True</code>, the gradients are multiplied by input.

### Integrated Gradients
Integrated Gradients are implemented in <code>integrated_gradients.py</code> as <code>ExplainerIG</code> class. The constructor has one more mandatory parameter <code>embeddings_tensor</code>, which is the extracted embeddings tensor on which a <code>gather</code> operation with <code>input_ids</code> can be performed - this is neccessary to generate a baseline. Then there is an optional parameter <code>baseline_tolerance</code>, which defaults to <code>0.05</code>. This parameter is used in baseline creation, defining how far from <code>0.5</code> the baseline can be - the optimal baseline for binary classification is specified as input, for which the model produces <code>0.5</code> output. This is different for multinomial models but can be easily modified in code.

Integrated Gradients tackle the issue of gradient saturation near input by interpolating samples from a neutral baseline (absence of information) to the input, creating a discrete approximation of the gradient function, which is then integrated. For the baseline creation an all-zero matrix is first evaluated and if the baseline is no good enough, it is then modified based on the gradients, so that it's sufficiently close to <code>0.5</code> (as defined by the <code>baseline_tolerance</code> parameter).

The <code>explain</code> method has several additional parameters:
<ul>
	<li><code>interpolation_steps</code> - How many interpolation steps should be used to calculate the integrated gradients. Usually 20 to 300 steps are sufficient. Defaults to 50.</li>
	<li><code>baseline</code> - Allows to pass a pre-generated baseline to be used. If <code>None</code>, baseline is generated automatically.</li>
	<li><code>x_input</code> - Whether or not to multiply the calculated integrated gradients by input. Defaults to <code>False</code></li>
</ul>

### SmoothGRAD
SmoothGRAD is implemented in <code>smoothgrad.py</code> as <code>ExplainerSmoothGRAD</code> class. the constructor has no mandatory parameters other than the usual <code>model</code>.

The <code>explain</code> method however takes several additional parameters:
<ul>
	<li><code>noise_size</code> - This parameter is used to generate samples around the input, descibed in detail below, but basically higher means more noise and lower means less noise. Default is <code>0.0125</code></li>
	<li><code>steps</code> - How many samples to generate</li>
	<li><code>x_inputs</code> - Whether or not to multiply the averaged gradients by input. Defaults to <code>False</code></li>
</ul>

SmoothGRAD tackles the issue of unstable gradients in the close area around the input trough adding more noise. The idea is, that we generate samples around the input, calculate their gradients and average them. This gives us an idea about how the model would behave for the explained input and very similar inputs. 

The samples are generated by adding uniform noise to the passed <code>input_embeds</code>. The standard deviation of the noise is calculated as <code>noise_size * (max(input_embeds) - min(input_embeds))</code>. 

### Attention
Attention mechanism in transformer models provides a way for us to see how much importance the model gave to each token in each layer. The attention mechanism can be different for different models, so the implementation may need some tweaks to work.

The method acquires attentions for the input by passing <code>output_attentions=True</code> to the <code>__call__</code> method of the model, so the model must deal with this internally and output attentions in shape <code>(sequence_length,)</code>, so one attention value for each token.

The attentions themselves provide no information about why the model was looking at a token - we have no way to differentiate between positive and negative attributions. For that reason we use gradients of the model output w.r.t. passed <code>input_embeds</code>, which gives us signs we can assign to the attention values. This creates a problem since the gradients are noisy, so we use only the sign of the gradients (<code>tf.sign(gradients)</code>) and we multiply the attentions by that.

The method requires no special parameters in the constructor, however there are some special parameters in the <code>explain</code> method:
<ul>
	<li><code>method</code> - Whether to use pure gradients (<code>'grad'</code>) or SmoothGRAD (<code>smoothgrad</code>). Defaults to <code>'grad'</code>. The smooth gradients might be able to improve the attributions.</li>
	<li><code>steps</code> - If <code>method='smoothgrad'</code>, specifies the number of samples.</li>
	<li><code>noise_size</code> - If <code>method='smoothgrad'</code>, specifies the noise size (for a detailed explanation, look at SmoothGRAD above).</li>
</ul>

### LIME
LIME is a model-agnostic method, which approximates the complex model's behavior aroudn the input trough a simple linear model, where the attributions can be calculated very simply. This is done by pertubing the input and seeing how the model output changes. These pairs of <code>input, result</code> are then fed into weighted linear regression, which trains the model and we can simply take the weights of the model as attributions.

The method is implemented in <code>lime.py</code> as <code>ExplainerLIME</code> class. The constructor has a required parameter <code>pad_token_embedding</code>, which is the embedding vector of a padding token. This is used for pertubation if <code>'tokens'</code> method is used (look below), otherwise anything like <code>None</code> can be passed.

The <code>explain</code> method takes several non-standard parameters:
<ul>
	<li><code>pertubation_method</code> - If <code>'tokens'</code> is passed, random tokens in the passed <code>input_embeds</code> are replaced by the <code>pad_token_embedding</code> vector. If <code>'noise'</code> is passed, the <code>input_embeds</code> are pertubed using the same method as SmoothGRAD. Defaults to <code>'tokens'</code>.</li>
	<li><code>sample_count</code> - How many samples to generate</li>
	<li><code>noise_size</code> - Used if <code>pertubation_method='noise'</code>. The same as in SmoothGRAD. Effectively defines how close to the input will the pertubed samples be with larger value meaning less similar samples will be created. Defaults to <code>0.01</code>.</li>
	<li><code>cover_prob</code> - Applies only when <code>pertubation_menthod='tokens'</code>. Probability of each token to be replaced with passed <code>pad_token_embedding</code>. Defaults to <code>0.125</code>.</li>
</ul> 

## Visualisation methods

All visualisation methods produce HTML, which can be opened in web browser. Visualisation is created given an instance of Explanation, which contains attributions for each input token, the input tokens and the classification result. The visualisation is also geared towards binary classification, but that can be easily changed. All the visualisation methods are implemented in <code>transformer-explainer/visualisers</code>.

The visualisers take into consideration the <code>&lt;p></code> and <code>&lt;/p></code> tags, which are not colored.

Both visualisation methods are implemented in <code>transformer_explainer/visualisers</code>.

The visualisers have a static method <code>produce_html</code>, which takes an <code>Explanation</code> instance as the only mandatory parameter. This method has a some tunable parameters:
<ul>
	<li><code>percentile</code> - This parameter specifies which extreme values will be cut-off. The percentile value is calculated from all attributions in absolute value and the attributions above the percentile value are zeroed. Default is 0.95</li>
	<li><code>word_processing</code> - This parameter determines how the attribution of a word is calculated in case the word consists of multiple tokens. The default is <code>'sum'</code> and the alternative is <code>'average'</code>.</li>
	<li><code>attr_only</code> - This parameter determines which attributions are visualised - the default is <code>'pos'</code>, which means only positive attributions are visualised, other options are <code>'neg'</code> for negative values only and <code>'both'</code> for both.</li>
</ul>

The default behaviour of the visualiser is, that the positive attributions are red and the negative are green. This behaviour right now requires a change in the <code>transformer_explainer/utils/utils.py</code> file, function <code>get_word_html</code>, where the value is compared to 0.

### Word visualiser
Class <code>VisualiserWordsHTML</code> parses the tokens into words - if a word is split into multiple tokens, the tokens (apart from the first one) are prefixed with <code>##</code>.

### Sentence visualiser
Class <code>VisualiserSentencesHTML</code> parses the text into words and then the words into sentences. This process is the same process the <code>VisualiserWordsHTML</code> uses. The sentence parsing is primitive and prone to errors. <code>VisualiserSentencesHTML</code>'s static method <code>produce_html</code> has some additional parameters:
<ul>
	<li><code>sentence_processing</code> - Determines how the sentence attributions are calculated - the default is <code>'average'</code>, which averages the attributions of the words, which the sentence consists of. The alternative is <code>'sum'</code>.</li>
	<li><code>sentence_limit</code> - This parameter enforces a limit on how many sentences can be visualised. The default is <code>'auto'</code>, which uses reference values that seemed appropriate and scales them based on the word and sentence counts. A specific number can be passed or <code>None</code> for no limit.</li>
	<li><code>double_sentence_limit_for_both</code>If both negative and positive attributions are to be visualised, this parameter can be set to either <code>True</code> (default) or <code>False</code>.</li>
</ul> 

## Usage examples

The <code>Explanation</code> instance can be serialized into JSON using <code>to_json</code> method, which returns a string, and deserialized using the static <code>Explanation.from_json(str)</code> method, whoch returns an <code>Explanation</code> instance. 

The <code>vis.py</code> file contains Python code which takes a file as an argument, reads the JSON representation of an <code>Explanation</code> from a file and produces HTML using <code>VisualiserSentencesHTML</code>.

The <code>main.py</code> file shows how an <code>Explanation</code> is created using Integrated Gradients. First, the model is created and its weights are loaded, then a tokenizer for the model is obtained. This preprocessing is specific for the model the library was created for. Tha main part is that the explanation is the created only using the <code>explain</code> method.