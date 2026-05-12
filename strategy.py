# 4h_KeltnerBreakout_12hTrend
# Hypothesis: Use 12h Keltner Channel breakout with 12h EMA20 trend filter and volume confirmation for 4h entries.
# Keltner adapts to volatility, reducing false breakouts in ranging markets. Trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation filters low-conviction moves. Target 20-50 trades/year for low fee drag.
# Works in bull markets (trend continuation) and bear markets (sharp reversals with volume).

name = "4h_KeltnerBreakout_12hTrend"
timeframe = "4h"
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

    # Get 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # 12h EMA20 for Keltner middle line and trend
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    # 12h ATR(10) for Keltner width
    tr12 = np.maximum(np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1])), np.abs(low_12h[1:] - close_12h[:-1]))
    tr12 = np.concatenate([[np.nan], tr12])
    atr10_12h = pd.Series(tr12).ewm(span=10, adjust=False, min_periods=10).mean().values
    keltner_upper_12h = ema20_12h + 2 * atr10_12h
    keltner_lower_12h = ema20_12h - 2 * atr10_12h

    # Use previous 12h bar's Keltner bands (only after close)
    keltner_upper_prev = np.roll(keltner_upper_12h, 1)
    keltner_lower_prev = np.roll(keltner_lower_12h, 1)
    keltner_upper_prev[0] = np.nan
    keltner_lower_prev[0] = np.nan

    # Align to 4h
    keltner_upper_aligned = align_htf_to_ltf(prices, df_12h, keltner_upper_prev)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_12h, keltner_lower_prev)
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)

    # Volume confirmation: volume > 2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above upper Keltner + above EMA20 + volume spike
            if (close[i] > keltner_upper_aligned[i] and 
                close[i] > ema20_12h_aligned[i] and 
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below lower Keltner + below EMA20 + volume spike
            elif (close[i] < keltner_lower_aligned[i] and 
                  close[i] < ema20_12h_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below lower Keltner OR below EMA20
            if close[i] < keltner_lower_aligned[i] or close[i] < ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above upper Keltner OR above EMA20
            if close[i] > keltner_upper_aligned[i] or close[i] > ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals