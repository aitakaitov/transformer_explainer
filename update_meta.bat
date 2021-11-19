ssh barticka@minos.zcu.cz
cd transformer/explanations
rm meta-*
rm -rf utils
rm -rf explainers
exit
scp -r "C:\Users\Vojtěch Bartička\PycharmProjects\transformer_explainer\meta-*" barticka@minos.zcu.cz:~/transformer/explanations
scp -r "C:\Users\Vojtěch Bartička\PycharmProjects\transformer_explainer\utils" barticka@minos.zcu.cz:~/transformer/explanations
scp -r "C:\Users\Vojtěch Bartička\PycharmProjects\transformer_explainer\explainers" barticka@minos.zcu.cz:~/transformer/explanations