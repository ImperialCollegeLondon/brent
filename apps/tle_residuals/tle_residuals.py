from apps.application import Application, ApplicationFactory


@ApplicationFactory.register("tle_residuals", "RIC residuals of a saved TLE fit")
class TLEResiduals(Application):

    @staticmethod
    def run(arguments):
        from ._tle_residuals import main

        main(
            arguments.input,
            arguments.output,
            arguments.dense_step_hours,
        )

    @classmethod
    def addArguments(cls, parser):
        parser.add_argument("-i", "--input", required=True,
                            help="Single-object tle_fit output directory")
        parser.add_argument("-o", "--output",
                            help=(
                                "CSV path (default: <fit directory>/"
                                "<run>_residuals.csv)"
                            ))
        parser.add_argument("--dense-step-hours", type=float, default=1.0,
                            help="Dense-grid timestep in hours (default: 1)")
