from app.plugin import AppPlugin
from .src.views import maneuverAnticipation

__plugin__ = "ManeuverAnticipation"
__version__ = "1.0.0"


class ManeuverAnticipation(AppPlugin):

    def setup(self):
        self.register_blueprint(maneuverAnticipation)
