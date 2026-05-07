#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: On 12h chart, enter long when price breaks above Camarilla R3 level with volume confirmation and daily trend alignment,
# enter short when price breaks below S3 level with volume confirmation and daily trend alignment.
# Uses 1d EMA34 for trend filter, volume spike confirmation, and exits on price reversing back below R3/above S3.
# Designed for low trade frequency (~15-30/year) to minimize fee decay and work in trending markets.
# Camarilla levels provide institutional support/resistance, effective in both bull and bear regimes.
# Timeframe: 12h, HTF: 1d for trend filter, volume confirmation on same timeframe.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Camarilla parameters (based on previous day)
    lookback = 1  # previous day's range
    
    # Calculate 1d EMA34 for trend filter (updated once per day)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1-day range for Camarilla levels (yesterday's high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    daily_range = high_1d - low_1d
    
    # Camarilla levels for today (based on yesterday's OHLC)
    close_prev = np.roll(close_1d, 1)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev[0] = close_1d[0]  # avoid NaN on first
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    
    # Camarilla R3 and S3 levels
    r3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    s3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (available after daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # start after EMA warmup
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + daily uptrend (price > EMA34)
            if close[i] > r3_aligned[i] and volume[i] > 1.8 * vol_ma[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume spike + daily downtrend (price < EMA34)
            elif close[i] < s3_aligned[i] and volume[i] > 1.8 * vol_ma[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes back below R3 (mean reversion)
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes back above S3 (mean reversion)
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals