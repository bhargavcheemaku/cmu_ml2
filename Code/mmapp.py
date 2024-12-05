from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "pong"})

@app.route('/simulateTourney', methods=['POST'])
def simulateTourney():
    try:
        data = request.get_json()
        mode = data.get('input')
        logging.debug(f"Evaluating with mode: {mode}")

        # Placeholder for core logic (update as necessary)
        result = simulate_tourney(curr_tourney, mode, best_model=best_model, predictors=predictors)
        logging.debug(f"Result: {result}")

        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in simulateTourney: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
