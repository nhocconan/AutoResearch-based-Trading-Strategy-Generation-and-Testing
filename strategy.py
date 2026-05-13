#!/usr/bin/env python3
# 6h_Keltner_Trend_RSI_Momentum
# Hypothesis: Price tends to revert to the mean within a Keltner channel during sideways markets, 
# but breaks out with momentum when the channel expands in trending conditions.
# Uses 60-period EMA as midline, 2x ATR for channel width, and 14-period RSI for momentum confirmation.
# In uptrends (price above EMA), go long when price touches lower band and RSI > 50.
# In downtrends (price below EMA), go short when price touches upper band and RSI < 50.
# Includes 1-day trend filter to ensure alignment with higher timeframe momentum.
# Designed for 6h timeframe to balance trade frequency and signal quality.
# Expected trades: 20-50 per year per symbol to minimize fee drag.

name = "6h_Keltner_Trend_RSI_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Keltner Channel: 60-period EMA as midline, 2x ATR for width
    ema60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    kc_upper = ema60 + 2 * atr
    kc_lower = ema60 - 2 * atr

    # RSI for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    # 1-day trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema60[i]) or 
            np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at lower Keltner band, uptrend, RSI > 50
            if close[i] <= kc_lower[i] and close[i] > ema60[i] and rsi[i] > 50 and ema50_1d_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at upper Keltner band, downtrend, RSI < 50
            elif close[i] >= kc_upper[i] and close[i] < ema60[i] and rsi[i] < 50 and ema50_1d_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above midline or RSI drops below 40
            if close[i] >= ema60[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below midline or RSI rises above 60
            if close[i] <= ema60[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals