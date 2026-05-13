from flask import Flask, request, jsonify
from workflow import graph
from langgraph.types import Command

app = Flask(__name__)


# Start Workflow Endpoint

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'Request body must be JSON'
            }), 400

        user_message = data.get('query')
        thread_id = data.get('thread_id', '1')

        if not user_message:
            return jsonify({
                'error': 'Missing required field: query'
            }), 400

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        events = graph.stream(
            {
                "messages": user_message
            },
            config,
            stream_mode="values"
        )

        final_response = None

        for event in events:

            # If graph is interrupted for human input
            if "__interrupt__" in event:
                interrupt_data = event["__interrupt__"]

                return jsonify({
                    "status": "waiting_for_human",
                    "thread_id": thread_id,
                    "interrupt": str(interrupt_data)
                })

            # Normal message flow
            if "messages" in event:
                final_response = event["messages"][-1].content

        return jsonify({
            "status": "completed",
            "thread_id": thread_id,
            "response": final_response
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500



# Resume Workflow Endpoint

@app.route('/api/human-response', methods=['POST'])
def human_response():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'Request body must be JSON'
            }), 400

        human_input = data.get("human_response")
        thread_id = data.get("thread_id", "1")

        if not human_input:
            return jsonify({
                'error': 'Missing required field: human_response'
            }), 400

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # Resume graph execution
        human_command = Command(
            resume={
                "data": human_input
            }
        )

        events = graph.stream(
            human_command,
            config,
            stream_mode="values"
        )

        final_response = None

        for event in events:

            if "messages" in event:
                final_response = event["messages"][-1].content

        return jsonify({
            "status": "completed",
            "thread_id": thread_id,
            "response": final_response
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
