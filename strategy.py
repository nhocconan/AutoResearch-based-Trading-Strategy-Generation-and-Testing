#!/usr/bin/env python3
# 4h_4H_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Use weekly trend filter (EMA50) with daily Camarilla R3/S3 breakout levels on 4h timeframe.
# Weekly EMA50 determines long/short bias, daily R3/S3 provides entry levels, volume confirms breakout.
# This approach reduces false signals by requiring alignment between weekly trend and daily breakout.
# Designed for low trade frequency (15-40 total trades over 4 years) with clear entry/exit rules to avoid overtrading.

name = "4h_4H_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Camarilla pivot levels: R3, S3, PP
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    r3_1d = close_1d + ((high_1d - low_1d) * 1.2500)
    s3_1d = close_1d - ((high_1d - low_1d) * 1.2500)
    pp_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align daily Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d.values)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d.values)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d.values)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + weekly uptrend (price above weekly EMA50) + volume spike
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + weekly downtrend (price below weekly EMA50) + volume spike
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP) or trend changes (price below weekly EMA50)
            if (close[i] <= pp_1d_aligned[i] or close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP) or trend changes (price above weekly EMA50)
            if (close[i] >= pp_1d_aligned[i] or close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals