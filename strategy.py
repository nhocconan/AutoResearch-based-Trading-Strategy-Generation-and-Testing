#!/usr/bin/env python3
# 1d_RSI_Overbought_Oversold_With_Volume_Confirmation
# Hypothesis: RSI extremes (below 30 for long, above 70 for short) combined with volume confirmation
# captures mean reversion in both bull and bear markets. Uses 1w EMA50 as trend filter to avoid
# trading against the higher timeframe trend. Target: 15-30 trades per year to minimize fee drag.

name = "1d_RSI_Overbought_Oversold_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold (<30) + price above 1w EMA50 + volume confirmation
            if (rsi[i] < 30 and 
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) + price below 1w EMA50 + volume confirmation
            elif (rsi[i] > 70 and 
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or trend change
            if rsi[i] > 70 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or trend change
            if rsi[i] < 30 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals