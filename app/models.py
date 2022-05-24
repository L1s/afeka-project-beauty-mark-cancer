from app.db import db


class AnalysisRequest(db.Model):
    __tablename__ = 'analysis_request'
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    image = db.Column(db.LargeBinary, nullable=False)
    gender = db.Column(db.String(length=20), nullable=False)
    age = db.Column(db.Float, nullable=False)
    result_label = db.Column(db.String(length=20), nullable=False)

    confirmed = db.Column(db.Boolean, nullable=False, default=False)
