# 4h_Keltner_Trend_Breakout_Volume
# Hypothesis: Breakouts from Keltner Channel (ATR-based) with 1d trend filter and volume spike capture momentum moves.
# Works in bull markets via breakouts and in bear via mean reversion touches of the middle line (EMA).
# Target: 20-40 trades/year per symbol.

name = "4h_Keltner_Trend_Breakout_Volume"
timeframe = "4h"
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

    # Get 1d data for trend filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate ATR for Keltner Channel (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values

    # EMA20 for Keltner middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Keltner Channel bounds
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above Keltner Upper + 1d uptrend + volume spike
            if close[i] > keltner_upper[i] and close[i] > ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 2:
                signals[i] = 0.30
                position = 1
            # SHORT: Close breaks below Keltner Lower + 1d downtrend + volume spike
            elif close[i] < keltner_lower[i] and close[i] < ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 2:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below EMA20 (middle line) or 1d trend turns down
            if close[i] < ema20[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Close crosses above EMA20 (middle line) or 1d trend turns up
            if close[i] > ema20[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals