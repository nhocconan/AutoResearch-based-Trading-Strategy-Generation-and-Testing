# 1d_Camarilla_Pivot_Volume_Trend
# Hypothesis: Camarilla pivot levels (R3/S3) on 1d chart, with volume confirmation and 1w trend filter, capture major reversals in both bull and bear markets.
# Breakout above R3 with weekly uptrend = long; breakdown below S3 with weekly downtrend = short.
# Uses weekly EMA for trend filter to reduce whipsaws and capture only strong moves.
# Target: 10-25 trades/year per symbol with disciplined risk management.

name = "1d_Camarilla_Pivot_Volume_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1w data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Close + 1.1 * (High - Low) * 1.1 / 2
    # S3 = Close - 1.1 * (High - Low) * 1.1 / 2
    pivot = (high + low + close) / 3
    price_range = high - low
    camarilla_r3 = close + 1.1 * price_range * 1.1 / 2
    camarilla_s3 = close - 1.1 * price_range * 1.1 / 2

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above Camarilla R3 + 1w uptrend + volume spike
            if close[i] > camarilla_r3[i] and close[i] > ema50_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Camarilla S3 + 1w downtrend + volume spike
            elif close[i] < camarilla_s3[i] and close[i] < ema50_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below Camarilla pivot or 1w trend turns down
            if close[i] < pivot[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above Camarilla pivot or 1w trend turns up
            if close[i] > pivot[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals