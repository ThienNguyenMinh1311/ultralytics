# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    try:
        # Check if Ray Tune session is active
        if get_context() is not None:
            metrics = trainer.metrics
            metrics["epoch"] = trainer.epoch
            session.report(metrics)
    except Exception:
        pass  # Silently ignore if Ray is not active


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
