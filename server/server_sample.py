from flask import Flask, request, jsonify
from flask_cors import CORS

import json

FlaskApp = Flask(__name__)
# CORS(FlaskApp, resources={r"/api/*": {"origins": ["http://localhost:5000", "*"]}})

@FlaskApp.route('/api/mission', methods=['GET'])
def get_telescope_position():
    date = request.args.get('date')
    telescope = request.args.get('telescope')
    if telescope == "4":
        with open(r"server\data.json", "r") as json_file:
            data = json.load(json_file)
        return jsonify(data)
    return jsonify({})

@FlaskApp.route('/', methods=['GET'])
def home():
    print("index")
    return jsonify("index")

FlaskApp.run(host="127.0.0.1", port=5050)



