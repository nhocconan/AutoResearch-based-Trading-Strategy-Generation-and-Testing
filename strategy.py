#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_Volume
# Hypothesis: Use 12-hour Camarilla pivot R3/S3 levels for breakout entries, filtered by 12-hour EMA50 trend and volume confirmation.
# Camarilla pivots identify key intraday support/resistance levels where breakouts often occur with momentum.
# The 12h EMA50 filter ensures trades align with the medium-term trend, reducing false breakouts.
# Volume confirmation adds conviction to breakout moves.
# Works in bull markets (follows upward breaks with bullish 12h trend) and bear markets (avoids upward breaks in bearish 12h trend, takes downward breaks).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_Volume"
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

    # Get 12h data for Camarilla pivot calculation and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate typical price for 12h
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    
    # Calculate Camarilla levels: R3, S3
    # Camarilla formulas: R3 = close + 1.1*(high - low)/2, S3 = close - 1.1*(high - low)/2
    # Using typical price as close equivalent for pivot calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_r3 = close_12h + 1.1 * (high_12h - low_12h) / 2
    camarilla_s3 = close_12h - 1.1 * (high_12h - low_12h) / 2
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Volume filter: >1.5x 20-period average on 4h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Camarilla R3 + price above 12h EMA50 (bullish trend) + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Camarilla S3 + price below 12h EMA50 (bearish trend) + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Camarilla S3 or price below 12h EMA50
            if (close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Camarilla R3 or price above 12h EMA50
            if (close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals