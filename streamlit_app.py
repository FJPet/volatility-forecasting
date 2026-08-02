import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pipeline import run_volatility_pipeline


# ==================================================
# Page configuration
# ==================================================
st.set_page_config(
    page_title="Explainable Volatility Forecasting",
    page_icon="📈",
    layout="wide"
)


# ==================================================
# Styling
# ==================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }
    .info-card {
        padding: 1.1rem;
        border-radius: 0.8rem;
        background-color: #f8f9fb;
        border: 1px solid #e6e8eb;
        margin-bottom: 1rem;
    }
    .small-muted {
        color: #666666;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("""
<div class="main-title">Explainable CNN Volatility Forecasting</div>

<div class="subtitle">
Forecast next-day stock volatility using intraday data and deep learning.
</div>

<div style="
padding:12px;
border-radius:10px;
background-color:#f8f9fb;
border:1px solid #e6e8eb;
margin-bottom:15px;
">

<b>What this app does:</b><br>
Forecasts next-day stock volatility from recent 1-minute intraday prices.

<br><br>
<b>Processing pipeline:</b><br>
1. The app first tries to download recent 1-minute market data from Yahoo Finance.<br>
2. If live data is unavailable, it automatically uses a stored demonstration snapshot for AAPL, MSFT or KO.<br>
3. Prices are restricted to regular US trading hours and converted into 1-minute log returns.<br>
4. Each trading day is transformed into a 380×380 waterfall image that preserves the timing, sign and magnitude of intraday returns.<br>
5. The pretrained convolutional neural network (CNN) processes the image and predicts next-day realized volatility.<br>
6. Grad-CAM highlights the image regions that had the strongest influence on the forecast.

<br><br>
<b>How to interpret results:</b><br>
• Values are next day’s <b>annualized volatility (%)</b><br>
• Higher values indicate greater expected market uncertainty<br>
• “Change” compares the prediction with the previous day’s realized volatility<br>
• A yellow data notice means the app is using a stored demonstration snapshot rather than live data

</div>
""", unsafe_allow_html=True)

# ==================================================
# Grad-CAM helper functions
# ==================================================
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

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        prediction = tf.convert_to_tensor(prediction)
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


def safe_model_predict(model, x):
    pred_raw = model.predict(x, verbose=0)

    if isinstance(pred_raw, (list, tuple)):
        pred_raw = pred_raw[0]

    return float(np.asarray(pred_raw).reshape(-1)[0])


def create_gradcam_overlay(model, img_array):
    x = np.expand_dims(img_array, axis=0).astype(np.float32)

    pred = safe_model_predict(model, x)
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

    return overlay, pred_annualized


# ==================================================
# Model loading
# ==================================================
MODEL_PATH = "model/trained_CNN.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ==================================================
# Sidebar
# ==================================================
st.sidebar.header("Forecast settings")

ticker_text = st.sidebar.text_input(
    "Tickers",
    value="AAPL, MSFT, KO",
    help="Enter one or more tickers separated by commas."
)

num_days = st.sidebar.slider(
    "Recent days of 1-minute data",
    min_value=5,
    max_value=14,
    value=7
)

run_button = st.sidebar.button(
    "Run forecast",
    type="primary",
    width="stretch"
)

st.sidebar.markdown("---")
st.sidebar.caption("Model output is reported as annualized realized volatility.")


# ==================================================
# Utility functions
# ==================================================
def compute_direction(last_row, results_df):
    previous_rows = results_df.dropna(subset=["actual_RV_annualized"])

    if previous_rows.empty:
        return np.nan, np.nan, "unknown"

    previous_vol = previous_rows.iloc[-1]["actual_RV_annualized"]
    diff = last_row["prediction_annualized"] - previous_vol

    if diff > 0:
        direction = "increase"
    elif diff < 0:
        direction = "decrease"
    else:
        direction = "unchanged"

    return previous_vol, diff, direction


def make_forecast_plot(results_df, ticker):
    plot_df = results_df.dropna(subset=["actual_RV_annualized"]).copy()

    if plot_df.empty:
        return None

    plot_df["predicts_for"] = pd.to_datetime(plot_df["predicts_for"])

    last_row = results_df.iloc[-1]
    next_pred = last_row["prediction_annualized"]

    last_known_date = plot_df["predicts_for"].max()
    next_date = last_known_date + pd.Timedelta(days=1)

    mse = np.mean(
        (plot_df["prediction_annualized"] - plot_df["actual_RV_annualized"]) ** 2
    )
    rmse = np.sqrt(mse)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        plot_df["predicts_for"],
        plot_df["actual_RV_annualized"],
        marker="o",
        linewidth=2,
        label="Actual RV"
    )

    ax.plot(
        plot_df["predicts_for"],
        plot_df["prediction_annualized"],
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Predicted RV"
    )

    ax.scatter(
        next_date,
        next_pred,
        s=130,
        label="Next-day forecast"
    )

    ax.plot(
        [plot_df["predicts_for"].iloc[-1], next_date],
        [plot_df["prediction_annualized"].iloc[-1], next_pred],
        linestyle="--",
        linewidth=1.8
    )

    ax.set_title(
        f"{ticker} annualized realized volatility forecast | RMSE: {rmse:.4f}%"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


# ==================================================
# Tabs
# ==================================================
tab_forecast, tab_model, tab_xai, tab_paper = st.tabs(
    [
        "Forecast Dashboard",
        "Model Idea",
        "Explainability",
        "Paper Context"
    ]
)


# ==================================================
# Forecast Dashboard
# ==================================================
with tab_forecast:
    st.markdown("## Forecast Dashboard")

    if not run_button:
        st.info("Enter tickers in the sidebar and click **Run forecast**.")
    else:
        tickers = [ticker.strip().upper() for ticker in ticker_text.split(",") if ticker.strip()]
        summary_rows = []

        for ticker in tickers:
            st.divider()
            st.markdown(f"### {ticker}")

            try:
                with st.spinner(f"Running forecast for {ticker}..."):
                    results_df, images_by_date, _ = run_volatility_pipeline(
                        ticker=ticker,
                        num_days=num_days,
                        model=model
                    )

                data_source = results_df.attrs.get("data_source", "unknown")

                if data_source == "live":
                    st.success(
                        "Data source: Live Yahoo Finance 1-minute data."
                    )
                elif data_source == "demo":
                    demo_dates = pd.to_datetime(
                        results_df["input_day"],
                        errors="coerce"
                    ).dropna()

                    if not demo_dates.empty:
                        demo_start = demo_dates.min().strftime("%Y-%m-%d")
                        demo_end = demo_dates.max().strftime("%Y-%m-%d")
                        date_text = f" Snapshot period: {demo_start} to {demo_end}."
                    else:
                        date_text = ""

                    st.warning(
                        "Live Yahoo Finance data is currently unavailable. "
                        "The forecast is being generated from a stored "
                        f"demonstration dataset.{date_text}"
                    )
                else:
                    st.info(
                        "The app could not identify whether live or stored data was used."
                    )

                last_row = results_df.iloc[-1]
                previous_vol, diff, direction = compute_direction(last_row, results_df)

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Next-day forecast",
                    f"{last_row['prediction_annualized']:.2f}%"
                )

                col2.metric(
                    "Previous realized volatility",
                    "N/A" if pd.isna(previous_vol) else f"{previous_vol:.2f}%"
                )

                col3.metric(
                    "Expected change",
                    "N/A" if pd.isna(diff) else f"{diff:+.2f} pp",
                    direction
                )

                summary_rows.append({
                    "ticker": ticker,
                    "input_day": last_row["input_day"],
                    "forecast_for": last_row["predicts_for"],
                    "next_day_prediction_annualized": last_row["prediction_annualized"],
                    "previous_actual_vol_annualized": previous_vol,
                    "difference_vs_previous_day": diff,
                    "direction": direction
                })

                with st.expander("Show detailed forecast table"):
                    st.dataframe(results_df, width="stretch")

                st.markdown("#### Forecast performance")

                fig = make_forecast_plot(results_df, ticker)

                if fig is not None:
                    st.pyplot(fig, width="stretch")
                else:
                    st.warning("Not enough actual values to create forecast plot.")

                st.markdown("#### Model input and explanation")

                latest_day = sorted(images_by_date.keys())[-1]
                latest_img = images_by_date[latest_day]

                img_col1, img_col2 = st.columns(2)

                with img_col1:
                    st.image(
                        latest_img,
                        caption=f"{ticker}_{latest_day} | CNN input waterfall image",
                        width="stretch"
                    )

                with img_col2:
                    try:
                        gradcam_overlay, gradcam_pred = create_gradcam_overlay(
                            model,
                            latest_img
                        )

                        st.image(
                            gradcam_overlay,
                            caption=(
                                f"{ticker}_{latest_day} | Grad-CAM overlay | "
                                f"Forecast: {gradcam_pred:.2f}%"
                            ),
                            width="stretch"
                        )
                    except Exception as gradcam_error:
                        st.warning(
                            f"Grad-CAM could not be generated for {ticker}: {gradcam_error}"
                        )

            except Exception as error:
                st.error(f"Error for {ticker}: {error}")

                summary_rows.append({
                    "ticker": ticker,
                    "input_day": None,
                    "forecast_for": None,
                    "next_day_prediction_annualized": np.nan,
                    "previous_actual_vol_annualized": np.nan,
                    "difference_vs_previous_day": np.nan,
                    "direction": "error"
                })

        st.divider()
        st.markdown("## Cross-stock summary")

        summary_df = pd.DataFrame(summary_rows)

        if not summary_df.empty:
            summary_df = summary_df.sort_values(
                by="next_day_prediction_annualized",
                ascending=False,
                na_position="last"
            )

        st.dataframe(summary_df, width="stretch")


# ==================================================
# Model Idea
# ==================================================
with tab_model:
    st.markdown("## Model Idea")

    st.markdown(
        """
        <div class="info-card">
        This dashboard forecasts <b>next-day realized volatility</b> using recent
        intraday stock data and a pretrained convolutional neural network.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    ### End-to-end pipeline

    **1. Obtain intraday market data**  
    The app requests recent 1-minute prices from Yahoo Finance. If the live
    request fails, a stored snapshot is used so the public demonstration
    remains functional.

    **2. Standardize the trading day**  
    Observations are converted to New York time, restricted to regular market
    hours and reindexed to a complete 1-minute grid. Missing prices are
    forward-filled.

    **3. Calculate intraday returns**  
    The app computes 1-minute logarithmic returns between 09:35 and 15:55,
    resulting in 380 return observations for a complete trading day.

    **4. Transform the time series into an image**  
    Consecutive returns are connected as vertical bars. Positive movements are
    shown in green and negative movements in red. The resulting 380×380 RGB
    waterfall image preserves the sequence and local structure of intraday
    movements.

    **5. Run the pretrained CNN**  
    The image is passed to a convolutional neural network trained to predict
    the following trading day’s realized volatility.

    **6. Annualize and evaluate the forecast**  
    Predictions and realized volatility are reported on an annualized scale.
    Where actual values are available, the dashboard also reports forecast
    errors and RMSE.

    **7. Explain the prediction with Grad-CAM**  
    Grad-CAM produces a heatmap showing which regions of the input image had
    the strongest influence on the CNN forecast.
    """)

    st.markdown("""
    ### Why image representations?

    The image representation preserves information that would be lost in simple daily aggregates:

    - intraday timing  
    - sign of returns  
    - volatility clustering  
    - local return patterns  
    - nonlinear market dynamics  
    """)


# ==================================================
# Explainability
# ==================================================
with tab_xai:
    st.markdown("## Explainability with Grad-CAM")

    st.markdown(
        """
        <div class="info-card">
        Grad-CAM highlights the regions of the CNN input image that were most relevant
        for the volatility forecast.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    In the dashboard:

    - the left image shows the original waterfall input  
    - the right image shows the Grad-CAM overlay  
    - warmer regions indicate stronger influence on the prediction  

    This helps answer:

    > Which intraday return patterns influenced the predicted volatility?
    """)

    st.markdown("""
    ### Interpretation

    The model may focus on:

    - intraday volatility clusters  
    - sharp return changes  
    - downside movements  
    - patterns around market open or close  
    - persistent return oscillations  

    This makes the forecast more transparent than a black-box prediction alone.
    """)


# ==================================================
# Paper Context
# ==================================================
with tab_paper:
    st.markdown("## Paper Context")

    st.markdown("""
    This dashboard implements and extends a research approach for volatility forecasting
    using image-based deep learning on intraday returns.

    The app turns the research workflow into an interactive prototype:

    - user enters tickers  
    - the system downloads recent intraday data  
    - the pretrained CNN generates forecasts  
    - the results are visualized  
    - Grad-CAM explains the latest prediction  
    """)

    st.markdown("### Reference")

    st.markdown("""
    **Betz, Niklas; Heil, Thomas L.A.; Peter, Franziska (2023)**  
    *Image-Based Deep Learning for Volatility Forecasting: A CNN–HAR Fusion Approach Using Intraday Returns*

    [SSRN Paper](https://ssrn.com/abstract=4584108)  
    [DOI Link](http://dx.doi.org/10.2139/ssrn.4584108)
    """)