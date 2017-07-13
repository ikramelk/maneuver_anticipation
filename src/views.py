from matplotlib.path import Path
from flask import Blueprint, render_template, current_app,request
import os


#from .algorithms.drivergraph import DriverGraph
import flask_plugins

from .algorithms.getdata import GetData
from .algorithms.clm import ManAnt
from flask_login import login_user, logout_user, login_required, current_user
import flask_plugins

from .algorithms.getdata import GetData
from .algorithms.clm import GOCLM
from flask_login import login_user, logout_user, login_required, current_user

maneuverAnticipation = Blueprint(
    "maneuverAnticipation",
    __name__,
    url_prefix='/ma',
    template_folder="../templates"
)



@maneuverAnticipation.route("/")
def index():
    return render_template("maneuver.html")

@maneuverAnticipation.route("/predictaction")
def predictaction():
    return render_template("predictaction.html")

'''@maneuverAnticipation.route('/getdata', methods=['GET'])
@login_required

def load_data():
	gd = GetData()
	gd.data()
	return render_template("maneuver.html")'''

@maneuverAnticipation.route('/generate', methods=['POST'])
def data_preprocessing():
    manant = ManAnt()
    manant.generateObservationsCLM()
    goclm = GOCLM()
    goclm.generateObservationsCLM()
    return render_template("maneuver.html")

@maneuverAnticipation.route('/predict', methods=['POST'])
def prediction():
    manant = ManAnt()
    manant.AIOhmmTrain()
    goclm = GOCLM()
    goclm.AIOhmmTrain()
    return render_template("maneuver.html")

@maneuverAnticipation.route('/upload', methods=['POST'])
def upload_file():
    fichier=request.files['file']
    manant = ManAnt()
    manant.generateObservationsCLMAction(fichier.filename)

    return render_template("predictaction2.html",action=manant.AIOhmmTrainAction())
    goclm = GOCLM()
    goclm.generateObservationsCLMAction(fichier.filename)

    return render_template("predictaction2.html",action=goclm.AIOhmmTrainAction())
    
