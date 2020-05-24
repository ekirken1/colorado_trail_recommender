from flask import Flask, render_template
app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trails')
def get_recommendation():
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
