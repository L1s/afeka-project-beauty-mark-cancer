from pathlib import Path

from flask import Flask


from app.analyser import init_analyser, get_analyser
from app.analysis import analysis_api
from app.db import init_db
from app.background import init_scheduler, scheduler
from app.main import main_blueprint
from model_utils.model_configurations import LOCAL_DIR

CURRENT_DIR = Path(__file__).parent
DB_PATH = LOCAL_DIR / 'app.sqlite'

app = Flask(__name__,
            static_url_path='/static',
            static_folder=str(CURRENT_DIR / 'static'),
            template_folder=str(CURRENT_DIR / 'templates'))


def retrain_model():
    print('Running retrain!!!!')
    with app.app_context():
        get_analyser().retrain()


def create_app():
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{str(DB_PATH)}'

    init_analyser(LOCAL_DIR / 'weights' / 'model_ourmodel.h5')

    # Register BLUEPRINTS
    app.register_blueprint(main_blueprint, url_prefix='/')
    app.register_blueprint(analysis_api, url_prefix='/api/analyse')

    init_db(app)
    init_scheduler(app)

    scheduler.add_job(retrain_model, 'cron', id='model-retrain', replace_existing=True,
                      day_of_week='sun')

    return app
