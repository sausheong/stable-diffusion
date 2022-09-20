from flask import Flask, request, render_template, redirect, session
from threading import Thread
from process import generate_image
import uuid
import os
import glob
import json
import time

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = b'stablediffusion_secret'

@app.get("/")
def main(images=None):
    images = filter(os.path.isfile, glob.glob("static/generated/*") )
    images = sorted(images, key=lambda x: os.path.getmtime(x), reverse=True)

    with open('generated.json') as json_file:
        images = json.load(json_file)
    images.sort(key=lambda x: x['timestamp'], reverse=True)
    return render_template("index.html", images=images)

@app.get("/view/<id>")
def view(id=None):
    return render_template("view.html", id=id, prompt=session["prompt"])

@app.get("/delete/<id>")
def delete(id=None):
    with open('generated.json') as json_file:
        images = json.load(json_file)

    images = [i for i in images if not (i['file'] == id)]

    with open("generated.json", "w") as outfile:
        json.dump(images, outfile)    

    return redirect("/")

@app.post("/gen")
def generate():
    prompt = request.form["prompt"]
    session["prompt"] = prompt
    id = str(uuid.uuid4())
    thread = Thread(target=generate_image, args=(prompt, id))
    thread.daemon = True
    thread.start()

    with open('generated.json') as json_file:
        images = json.load(json_file)
    images.append({"timestamp": time.time(), "file": f"{id}.png", "prompt": f"{prompt}"})
    with open("generated.json", "w") as outfile:
        json.dump(images, outfile)    

    return redirect(f"/view/{id}")