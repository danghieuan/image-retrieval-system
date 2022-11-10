import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image as kimage



app = Flask(__name__)


fe = FeatureExtractor()

# Define Cosine Similarity function
def cosine_similarity(query, X):
    norm_2_query = np.sqrt(np.sum(query*query))
    norm_2_X = np.sqrt(np.sum(X*X, axis=-1))
    return np.sum(query*X, axis=-1)/(norm_2_query*norm_2_X)

# Define result of retrieving image
def retrieval_images(query_vector, imgs_feature):
    values = cosine_similarity(query_vector, imgs_feature) # caculate cosine similarity between query and features in database
    id_s = np.argsort(-values)[:20] # Getting top 20 nearest results

    return [(round(values[id], 2), paths_feature[id]) for id in id_s]

root_fearure_path = "./static/feature_database/concat_all_feature.npz"


data = np.load(root_fearure_path)
paths_feature = data["array_1"]
imgs_feature = data["array_2"]

@app.route("/", methods = ["GET", "POST"])

def index():
    if request.method == "POST":
        file = request.files["query_img"]

        # Save query image from Flask server into static/image_uploaded/
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/image_uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        
        # Load query image and FeatureExtractor
        query = kimage.load_img(uploaded_img_path, target_size=(224, 224))
        query = kimage.img_to_array(query, dtype = np.float32)
        query_vector = fe.extract(query[None, :])  

        # retrieval_images
        scores = retrieval_images(query_vector, imgs_feature)

        return render_template("index.html", query_path = uploaded_img_path, scores = scores)

    return render_template("index.html")


if __name__ == "__main__":
    app.run()