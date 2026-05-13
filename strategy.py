#!/usr/bin/env python3
# 1d_WeeklyTrend_Following_With_Volume
# Hypothesis: Follow weekly trend (EMA50) on daily timeframe with volume confirmation.
# Enter long when daily close > weekly EMA50 and volume > 2x 5-day average volume.
# Enter short when daily close < weekly EMA50 and volume > 2x 5-day average volume.
# Exit when price crosses back over weekly EMA50.
# Weekly trend filter reduces whipsaw, volume ensures institutional participation.
# Works in bull (trend following) and bear (trend following short).
# Low frequency due to weekly trend filter and volume confirmation.

name = "1d_WeeklyTrend_Following_With_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 2.0 * 5-day average volume (1 week worth at 1d)
    vol_ma_5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    volume_filter = volume > 2.0 * vol_ma_5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > weekly EMA50 + volume filter
            if close[i] > ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < weekly EMA50 + volume filter
            elif close[i] < ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly EMA50
            if close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly EMA50
            if close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals