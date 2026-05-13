#!/usr/bin/env python3
# 1h_SwingReversal_4hTrend_Filter
# Hypothesis: Trade 1-hour reversals from intraday swing points, filtered by 4-hour trend direction.
# Long when price rejects a 1-hour swing low during 4h uptrend with volume confirmation.
# Short when price rejects a 1-hour swing high during 4h downtrend with volume confirmation.
# Exit on opposite swing rejection or trend flip.
# Uses 4h trend filter to avoid counter-trend whipsaws, targeting 15-35 trades/year per symbol.
# Designed to work in both bull (trend-following reversals) and bear (counter-trend bounces) markets.

name = "1h_SwingReversal_4hTrend_Filter"
timeframe = "1h"
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

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)

    # 1-hour swing points: swing high = high > prev 3 and next 3 highs
    # swing low = low < prev 3 and next 3 lows
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(3, n - 3):
        if (high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i-3] and
            high[i] > high[i+1] and high[i] > high[i+2] and high[i] > high[i+3]):
            swing_high[i] = True
        if (low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i-3] and
            low[i] < low[i+1] and low[i] < low[i+2] and low[i] < low[i+3]):
            swing_low[i] = True

    # Volume filter: >1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):
        # Skip if any required value is NaN
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price rejects swing low (close > swing low high) in 4h uptrend + volume spike
            if swing_low[i] and close[i] > low[i]:  # rejected swing low
                if close[i] > ema_50_4h_aligned[i] and volume[i] > vol_avg_20[i] * 1.3:
                    signals[i] = 0.20
                    position = 1
            # SHORT: Price rejects swing high (close < swing high low) in 4h downtrend + volume spike
            elif swing_high[i] and close[i] < high[i]:  # rejected swing high
                if close[i] < ema_50_4h_aligned[i] and volume[i] > vol_avg_20[i] * 1.3:
                    signals[i] = -0.20
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price rejects swing high or trend turns down
            if swing_high[i] and close[i] < high[i]:  # rejected swing high
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_50_4h_aligned[i]:  # trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price rejects swing low or trend turns up
            if swing_low[i] and close[i] > low[i]:  # rejected swing low
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_50_4h_aligned[i]:  # trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals