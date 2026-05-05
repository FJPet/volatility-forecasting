from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from pipeline import run_volatility_pipeline


app = FastAPI(title="CNN Volatility Forecast API")

model_path = "model/trained_CNN.keras"
model = tf.keras.models.load_model(model_path)


class ForecastRequest(BaseModel):
    tickers: list[str]
    num_days: int = 7


# ==================================================
# Helper functions for image responses
# ==================================================
def fig_to_png_response(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise RuntimeError("No Conv2D layer found.")


def compute_gradcam(model, image_b1, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, prediction = grad_model(image_b1, training=False)
        loss = prediction[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise RuntimeError("Gradients are None.")

    grads = grads[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# ==================================================
# Home endpoint
# ==================================================
@app.get("/")
def home():
    return {"message": "CNN Volatility Forecast API is running"}


# ==================================================
# Forecast endpoint: JSON output
# ==================================================
@app.post("/forecast")
def forecast(request: ForecastRequest):
    summary = []

    for ticker in request.tickers:
        ticker = ticker.upper()

        try:
            results_df, _, _ = run_volatility_pipeline(
                ticker=ticker,
                num_days=request.num_days,
                model=model
            )

            last_row = results_df.iloc[-1]
            previous_rows = results_df.dropna(subset=["actual_RV_annualized"])

            if not previous_rows.empty:
                previous_vol = previous_rows.iloc[-1]["actual_RV_annualized"]
                difference = last_row["prediction_annualized"] - previous_vol

                if difference > 0:
                    direction = "increase"
                elif difference < 0:
                    direction = "decrease"
                else:
                    direction = "unchanged"
            else:
                previous_vol = np.nan
                difference = np.nan
                direction = "unknown"

            summary.append({
                "ticker": ticker,
                "input_day": str(last_row["input_day"]),
                "forecast_for": str(last_row["predicts_for"]),
                "next_day_prediction_annualized": float(last_row["prediction_annualized"]),
                "previous_actual_vol_annualized": None if np.isnan(previous_vol) else float(previous_vol),
                "difference_vs_previous_day": None if np.isnan(difference) else float(difference),
                "direction": direction
            })

        except Exception as e:
            summary.append({
                "ticker": ticker,
                "error": str(e)
            })

    return {"summary": summary}


# ==================================================
# Plot endpoint: actual vs prediction + forecast
# ==================================================
@app.get("/plot/{ticker}")
def plot_forecast(ticker: str, num_days: int = 7):
    ticker = ticker.upper()

    results_df, _, _ = run_volatility_pipeline(
        ticker=ticker,
        num_days=num_days,
        model=model
    )

    plot_df = results_df.dropna(subset=["actual_RV_annualized"]).copy()

    if plot_df.empty:
        raise ValueError(f"Not enough actual realized volatility values for {ticker}")

    plot_df["predicts_for"] = pd.to_datetime(plot_df["predicts_for"])

    last_row = results_df.iloc[-1]
    next_pred = last_row["prediction_annualized"]

    last_known_date = plot_df["predicts_for"].max()
    next_date = last_known_date + pd.Timedelta(days=1)

    mse = np.mean(
        (plot_df["prediction_annualized"] - plot_df["actual_RV_annualized"]) ** 2
    )
    rmse = np.sqrt(mse)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        plot_df["predicts_for"],
        plot_df["actual_RV_annualized"],
        marker="o",
        label="Actual RV annualized"
    )

    ax.plot(
        plot_df["predicts_for"],
        plot_df["prediction_annualized"],
        marker="o",
        linestyle="--",
        label="Predicted RV annualized"
    )

    ax.scatter(
        next_date,
        next_pred,
        s=100,
        color="orange",
        label="Next-day forecast"
    )

    ax.plot(
        [plot_df["predicts_for"].iloc[-1], next_date],
        [plot_df["prediction_annualized"].iloc[-1], next_pred],
        linestyle="--",
        color="orange"
    )

    ax.set_title(
        f"{ticker} – Annualized Realized Volatility Forecast\n"
        f"RMSE annualized: {rmse:.4f}%"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility (%)")
    ax.legend()
    ax.grid(True)

    fig.autofmt_xdate()
    fig.tight_layout()

    return fig_to_png_response(fig)


# ==================================================
# Waterfall endpoint: CNN input image
# ==================================================
@app.get("/waterfall/{ticker}")
def waterfall_image(ticker: str, num_days: int = 7):
    ticker = ticker.upper()

    _, images_by_date, _ = run_volatility_pipeline(
        ticker=ticker,
        num_days=num_days,
        model=model
    )

    latest_day = sorted(images_by_date.keys())[-1]
    img = images_by_date[latest_day]

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    ax.imshow(img)
    ax.set_title(
        f"{ticker}_{latest_day} | CNN input waterfall image",
        color="white"
    )
    ax.axis("off")
    fig.tight_layout()

    return fig_to_png_response(fig)


# ==================================================
# Grad-CAM endpoint: explanation image
# ==================================================
@app.get("/gradcam/{ticker}")
def gradcam_image(ticker: str, num_days: int = 7):
    ticker = ticker.upper()

    _, images_by_date, _ = run_volatility_pipeline(
        ticker=ticker,
        num_days=num_days,
        model=model
    )

    latest_day = sorted(images_by_date.keys())[-1]
    img_array = images_by_date[latest_day]

    x = np.expand_dims(img_array, axis=0).astype(np.float32)

    pred = float(model.predict(x, verbose=0)[0][0])
    pred_annualized = (pred / 100) * np.sqrt(252)

    last_conv_layer = get_last_conv_layer_name(model)
    heatmap = compute_gradcam(model, x, last_conv_layer)

    sample_img = img_array.astype(np.float32)
    sample_img_vis = sample_img / 255.0 if sample_img.max() > 1 else sample_img

    heatmap_resized = cv2.resize(
        heatmap,
        (sample_img_vis.shape[1], sample_img_vis.shape[0])
    )

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )

    heatmap_rgb = heatmap_colored[..., ::-1] / 255.0

    overlay = 0.5 * heatmap_rgb + 0.5 * sample_img_vis
    overlay = np.clip(overlay, 0, 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(overlay)
    ax.set_title(
        f"{ticker}_{latest_day} | Grad-CAM\n"
        f"Predicted annualized RV: {pred_annualized:.2f}%"
    )
    ax.axis("off")
    fig.tight_layout()

    return fig_to_png_response(fig)