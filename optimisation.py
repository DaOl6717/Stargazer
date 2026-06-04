import optuna
from processing import process_image
from scoring import image_score, neutralise_color

def optimise_parameters(image, n_trials=50):
    def objective(trial):
        params = {
            "stretch": trial.suggest_float("stretch", 1.0, 2.0),
            "denoise": trial.suggest_float("denoise", 0.5, 2.0),
            "contrast": trial.suggest_float("contrast", 1.0, 1.8),
            "sharpen": trial.suggest_float("sharpen", 0.0, 1.0),
            "gamma": trial.suggest_float("gamma", 0.5, 2.0),
            "r_gain": ..., #trial.suggest_float("r_gain", 0.9, 1.1),
            "g_gain": ..., #trial.suggest_float("g_gain", 0.9, 1.1),
            "b_gain": ..., #trial.suggest_float("b_gain", 0.9, 1.1),
        }

        processed = process_image(image, params)
        processed = neutralise_color(processed)
        score = image_score(processed)

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params