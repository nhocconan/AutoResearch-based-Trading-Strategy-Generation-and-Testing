#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = df_1d['close'].shift(1).values
    
    # Handle first value for previous close
    prev_close = np.zeros_like(close_1d)
    prev_close[0] = close_1d[0]
    prev_close[1:] = close_1d[:-1]
    
    # Camarilla calculation
    range_1d = high_1d - low_1d
    R1 = close_1d + (range_1d * 1.1 / 12)
    S1 = close_1d - (range_1d * 1.1 / 12)
    R3 = close_1d + (range_1d * 1.1 / 4)
    S3 = close_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price crosses above R1 + uptrend + volume spike
            if (close[i] > r1 and 
                close[i] > ema34_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below S1 + downtrend + volume spike
            elif (close[i] < s1 and 
                  close[i] < ema34_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1 OR trend turns down
            if (close[i] < s1 or close[i] < ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R1 OR trend turns up
            if (close[i] > r1 or close[i] > ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals