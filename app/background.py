from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

from app.db import db

scheduler = BackgroundScheduler()


def init_scheduler(app):
    with app.app_context():
        scheduler.configure(jobstores={
            'default': SQLAlchemyJobStore(engine=db.engine)
        })
        scheduler.start()
