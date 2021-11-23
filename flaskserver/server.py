from flask import Flask
from flask import render_template, make_response

app = Flask(__name__)

# pass variables trough session dict

with open("templates/mainpage.html", "r") as f:
    mainpage_html = f.read()


@app.route("/")
def main_page():
    return mainpage_html


@app.route("/presentation", methods=['POST'])
def presentation_page():
    return make_response(render_template("templates/presentation.html"))
