import io
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
    test_size = StringField('Test size (ignore if No)', validators=[DataRequired()], default='0.2')
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


class TrainModel(FlaskForm):
    submit = SubmitField('Train model')


class Data:
    X_train = None
    X_test = None
    y_train = None
    y_test = None


class Model:
    model = None


data_class = Data()
model = Model()


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
                model.model = RandomForestMSE(n_estimators=n_trees,
                                        max_depth=max_depth,
                                        feature_subsample_size=features_size)
            else:
                model.model = GradientBoostingMSE(n_estimators=n_trees,
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
            tmp = pd.read_csv(stream)
            try:
                tmp = tmp.drop(columns=['id'])
            except:
                pass
            data_class.X_train = tmp.drop(columns=['price']).to_numpy()
            data_class.y_train = tmp['price'].to_numpy()

            stream = io.StringIO(train_test_form.file_path_test.data.stream.read().decode("UTF8"), newline=None)
            tmp = pd.read_csv(stream)
            try:
                tmp= tmp.drop(columns=['id'])
            except:
                pass
            data_class.X_test = tmp.drop(columns=['price']).to_numpy()
            data_class.y_test = tmp['price'].to_numpy()

            return redirect(url_for('train_model'))

        return render_template('from_form.html', form=train_test_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/load_data', methods=['POST', 'GET'])
def data():
    try:
        data_form = LoadAllData()

        if data_form.validate_on_submit():
            stream = io.StringIO(data_form.file_path.data.stream.read().decode("UTF8"), newline=None)
            data_class.X_train = pd.read_csv(stream)
            try:
                data_class.X_train = data_class.X_train.drop(columns=['id'])
            except:
                pass
            tmp = data_class.X_train
            data_class.X_train = tmp.drop(columns=['price']).to_numpy()
            data_class.y_train = tmp['price'].to_numpy()

            roadmap = data_form.roadmap.data == 'Yes'
            if roadmap:
                test_size = float(data_form.test_size.data)
                if not (0 <= test_size <= 1):
                    raise ValueError("Incorrect test_size")
                data_train, data_test, y_train, y_test = train_test_split(data_class.X_train, data_class.y_train, test_size=test_size)
                data_class.X_train = data_train
                data_class.X_test = data_test
                data_class.y_train = y_train
                data_class.y_test = y_test
            return redirect(url_for('train_model'))

        return render_template('from_form.html', form=data_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/train_model', methods=['POST', 'GET'])
def train_model():
    trained = False
    train_rmse = 'No info'
    test_rmse = 'No info'
    inference_info = ''
    try:
        train_form = TrainModel()

        if train_form.validate_on_submit():
            X_train = data_class.X_train
            y_train = data_class.y_train
            model.model.fit(X_train, y_train)
            y_pred = model.model.predict(X_train)
            train_rmse = mean_squared_error(y_train, y_pred, squared=False)
            if data_class.X_test is not None:
                y_pred = model.model.predict(data_class.X_test)
                test_rmse = mean_squared_error(data_class.y_test, y_pred, squared=False)
            trained = True
        if trained:
            inference_info = 'Go to inference model'
        return render_template('train_model.html', form=train_form, train_rmse=train_rmse, test_rmse=test_rmse, inference_info=inference_info)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/inference', methods=['POST', 'GET'])
def inference():
    pass