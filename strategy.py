#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
# Hypothesis: Breakout at Camarilla R1/S1 levels with 12h EMA trend filter and volume confirmation.
# Works in bull/bear markets by only taking breakouts in the direction of the 12h trend.
# Uses tight entry conditions to limit trades (~25-40/year) and avoid fee drag.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
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

    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Calculate 1d data for Camarilla levels (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Previous day's OHLC (Camarilla uses previous day)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + (prev_high - prev_low) * 1.0833
    S1 = prev_close - (prev_high - prev_low) * 1.0833
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 + above 12h EMA50 (uptrend) + volume spike
            if (close[i] > R1_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 + below 12h EMA50 (downtrend) + volume spike
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 (reversal) or below 12h EMA50 (trend change)
            if (close[i] < S1_aligned[i] or close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 (reversal) or above 12h EMA50 (trend change)
            if (close[i] > R1_aligned[i] or close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals