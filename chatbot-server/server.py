import os
from flask import Flask, request
from flask_cors import CORS
from lib.output_seq2seq import get_answer

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.run(host='0.0.0.0', port=5000)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )
    CORS(app)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    @app.route('/chat', methods=('POST',))
    def chat():
        body = request.get_json()
        if body['q'] is None:
            return 'Please assign q', 400
        print(body['q'])
        sentence = get_answer(body['q'])
        return sentence

    return app
