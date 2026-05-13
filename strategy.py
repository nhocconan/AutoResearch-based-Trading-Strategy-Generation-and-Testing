#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) from daily data with 1-week EMA trend filter and volume confirmation.
# Camarilla levels provide high-probability reversal/breakout zones; weekly EMA ensures alignment with higher timeframe trend.
# Works in bull markets (buying S3 breaks in uptrend) and bear markets (selling R3 breaks in downtrend).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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

    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (R3, S3) from previous day's OHLC
    # Camarilla formulas: R3 = close + 1.1*(high-low)*1.1/2, S3 = close - 1.1*(high-low)*1.1/2
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    r3 = prev_close + 1.1 * prev_range * 1.1 / 2
    s3 = prev_close - 1.1 * prev_range * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks below S3 (support) + price above weekly EMA (bullish trend) + volume spike
            if (close[i] < s3_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks above R3 (resistance) + price below weekly EMA (bearish trend) + volume spike
            elif (close[i] > r3_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks above R3 (resistance) or price below weekly EMA
            if (close[i] > r3_aligned[i] or close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks below S3 (support) or price above weekly EMA
            if (close[i] < s3_aligned[i] or close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals