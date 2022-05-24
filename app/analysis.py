from http import HTTPStatus
from typing import List

from PIL import Image
from flask import Blueprint, jsonify, request, abort
from marshmallow import Schema, fields, pre_dump
from tensorflow.keras.preprocessing import image

from app.analyser import get_analyser, ImageMeta
from app.db import db
from app.models import AnalysisRequest
from model_utils.model_configurations import IMAGE_SIZE
# from app import IMAGE_SIZE


analysis_api = Blueprint('analysis_api', __name__)


IMAGE_EXTENSIONS = ['jpg', 'jpeg']


class AnalysisSchema(Schema):
    id = fields.Integer(dump_only=True)
    name = fields.Str(dump_only=True)
    category = fields.Str(dump_only=True)
    value = fields.Str(dump_only=True)
    description = fields.Str(dump_only=True)

    @pre_dump
    def pre_dump(self, data, many):
        data, data_id = data
        return {
            'id': data_id,
            'name': data[0],
            'category': data[1],
            'value': int(data[2] * 100),
            'description': data[3],
        }


def allowed_file(filename: str, extensions: List[str] = None) -> bool:
    if extensions is None:
        return True

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


def get_file(name: str, extensions: List[str] = None):
    if name not in request.files:
        abort(HTTPStatus.BAD_REQUEST)
        return None

    file = request.files[name]
    if file.filename == '':
        abort(HTTPStatus.BAD_REQUEST)
        return None
    if not allowed_file(file.filename, extensions):
        abort(HTTPStatus.BAD_REQUEST)
        return None

    return file


def get_meta(name: str):
    if name not in request.form:
        abort(HTTPStatus.BAD_REQUEST)
        return None

    return request.form[name]


@analysis_api.route('/', methods=["POST"])
def analyse():
    image_file = get_file('image', extensions=IMAGE_EXTENSIONS)
    meta = ImageMeta(get_meta('gender'), float(get_meta('age')))

    analysis_request = AnalysisRequest()
    analysis_request.gender = meta.gender
    analysis_request.age = meta.age

    image_file.seek(0)
    with Image.open(image_file) as pil_image:
        pil_image = pil_image.resize(IMAGE_SIZE)
        arrimg = image.img_to_array(pil_image)

        analysis_request.image = arrimg.tobytes()

        analyser = get_analyser()
        result = analyser.analyse(arrimg, meta)

        analysis_request.result_label = result.name
        db.session.add(analysis_request)
        db.session.commit()

        schema = AnalysisSchema().dump((result, analysis_request.id))
        return jsonify(schema), HTTPStatus.OK


@analysis_api.route('/confirm/<int:request_id>/', methods=["POST"])
def analyse_confirm(request_id):
    analysis_request = db.session.query(AnalysisRequest)\
        .filter(AnalysisRequest.id != request_id)\
        .first_or_404()

    analysis_request.confirmed = True
    db.session.commit()

    return '', HTTPStatus.NO_CONTENT
