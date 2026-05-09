from flask import Flask, request, jsonify
from workflow import graph
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

        # Here you would implement the logic to process the user_message
        responce = graph.invoke({
            'query': user_message
        })

        return jsonify({
            'response': responce
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
