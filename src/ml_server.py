import io
import pandas as pd

from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, FileField, SelectField

from ensembles import GradientBoostingMSE, RandomForestMSE


RANDOM_FOREST_NAME = 'Random Forest MSE'
GRADIENT_BOOSTING_NAME = 'Gradient Boosting MSE'


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)

class ChooseParameters(FlaskForm):
    roadmap_list = ["Yes", "No"]
    roadmap = SelectField('Do you have a breakdown of the data into training and test ones?', choices=roadmap_list)
    model_list = [RANDOM_FOREST_NAME, GRADIENT_BOOSTING_NAME]
    model = SelectField('Select model', choices=model_list)
    n_trees = StringField('number of trees in ensemble', validators=[DataRequired()], default='100')
    features_size = StringField('feature subsample size', validators=[DataRequired()], default='0.7')
    max_depth = StringField('max depth', validators=[DataRequired()], default='6')
    learning_rate = StringField('learning_rate (will be ignored for Random Forest)', validators=[DataRequired()], default='0.05')
    submit = SubmitField('Load parameters and continue')


class LoadAllData(FlaskForm):
    roadmap_list = ["Yes", "No"]
    roadmap = SelectField('Do you want to divide the available data into training and test samples?', choices=roadmap_list)
    file_path = FileField('Load the dataset', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV format only!')
    ])
    submit = SubmitField('Load Data')


class LoadTrainTestData(FlaskForm):
    file_path_train = FileField('Load the train dataset', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV format only!')
    ])
    file_path_test = FileField('Load the test dataset', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV format only!')
    ])
    submit = SubmitField('Load Data')

class Data():
    train = None
    test = None

data = Data()
model = None


@app.route('/', methods=['POST', 'GET'])
def init():
    try:
        init_form = ChooseParameters()

        if init_form.validate_on_submit():
            roadmap = init_form.roadmap.data == 'Yes'
            n_trees = int(init_form.n_trees.data)
            features_size = float(init_form.features_size.data)
            max_depth = int(init_form.max_depth.data)
            learning_rate = float(init_form.learning_rate.data)
            if init_form.model.data == RANDOM_FOREST_NAME:
                model = RandomForestMSE(n_estimators=n_trees,
                                        max_depth=max_depth,
                                        feature_subsample_size=features_size)
            else:
                model = GradientBoostingMSE(n_estimators=n_trees,
                                            max_depth=max_depth,
                                            feature_subsample_size=features_size,
                                            learning_rate=learning_rate)
            if roadmap:
                return redirect(url_for('train_test'))
            return redirect(url_for('data'))

        return render_template('from_form.html', form=init_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/train_test_data', methods=['POST', 'GET'])
def train_test():
    try:
        train_test_form = LoadTrainTestData()

        if train_test_form.validate_on_submit():
            stream = io.StringIO(train_test_form.file_path_train.data.stream.read().decode("UTF8"), newline=None)
            data.train = pd.read_csv(stream)
            try:
                data.train = data.train.drop(columns=['id'])
            except:
                pass
            data.train = data.train.to_numpy()

            stream = io.StringIO(train_test_form.file_path_test.data.stream.read().decode("UTF8"), newline=None)
            data.test = pd.read_csv(stream)
            try:
                data.test = data.test.drop(columns=['id'])
            except:
                pass
            data.test = data.test.to_numpy()

            return redirect(url_for('train_model'))

        return render_template('from_form.html', form=train_test_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/load_data', methods=['POST', 'GET'])
def data():
    try:
        data_form = LoadAllData()

        if data_form.validate_on_submit():
            pass
        return render_template('from_form.html', form=data_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

