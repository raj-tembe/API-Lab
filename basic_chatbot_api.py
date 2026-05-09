from flask import Flask, request, jsonify
from llmchain import chain

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:

        data = request.get_json()

        # Validate request body
        if not data:
            return jsonify({
                'error': 'Request body must be JSON'
            }), 400

        # Validate required field
        user_message = data.get('query')

        if not user_message:
            return jsonify({
                'error': 'Missing required field: query'
            }), 400

        # Invoke LLM chain
        response = chain.invoke({
            'query': user_message
            })

        return jsonify({
            'response': response
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
