# 1d_Weekly_Camarilla_R3S3_Breakout_Trend_Volume
# Hypothesis: On daily timeframe, breakouts beyond weekly Camarilla R3/S3 levels with weekly trend alignment (EMA34) and volume spike capture strong momentum moves. Works in bull (breakouts continue) and bear (breakdowns continue) by following the weekly trend. Volume spike filters false breakouts. Weekly timeframe reduces noise, daily provides timely entries. Target: 20-50 trades over 4 years.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate weekly Camarilla levels from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_shift = df_1w['close'].shift(1).values
    high_1w_shift = df_1w['high'].shift(1).values
    low_1w_shift = df_1w['low'].shift(1).values
    
    # Calculate pivot and Camarilla levels using previous week's data
    pivot = (high_1w_shift + low_1w_shift + close_1w_shift) / 3
    range_ = high_1w_shift - low_1w_shift
    r3 = close_1w_shift + (range_ * 1.1 / 4)
    s3 = close_1w_shift - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above S3 + uptrend + volume spike
            if (close[i] > s3_val and 
                close[i] > ema34_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below R3 + downtrend + volume spike
            elif (close[i] < r3_val and 
                  close[i] < ema34_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR trend turns down
            if (close[i] < s3_val or close[i] < ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR trend turns up
            if (close[i] > r3_val or close[i] > ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals