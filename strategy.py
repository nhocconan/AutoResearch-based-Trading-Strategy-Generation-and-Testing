#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, trend filter, and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate previous 1d Camarilla levels
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Avoid division by zero in case high == low
    range_prev = high_prev - low_prev
    range_prev = np.where(range_prev == 0, 1e-10, range_prev)
    
    # Camarilla levels for previous day
    R3 = close_prev + 1.1 * range_prev * 1.125 / 4  # close + 1.1*range*1.125/4
    S3 = close_prev - 1.1 * range_prev * 1.125 / 4  # close - 1.1*range*1.125/4
    R4 = close_prev + 1.1 * range_prev * 1.5       # close + 1.1*range*1.5/2
    S4 = close_prev - 1.1 * range_prev * 1.5       # close - 1.1*range*1.5/2
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or
            np.isnan(S4_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34 = ema34_1d_aligned[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        r4 = R4_aligned[i]
        s4 = S4_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike, in uptrend (price > EMA34)
            if close[i] > r3 and vol_spike and close[i] > ema34:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike, in downtrend (price < EMA34)
            elif close[i] < s3 and vol_spike and close[i] < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 (failure) or trend reverses
            if close[i] < s3 or close[i] < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 (failure) or trend reverses
            if close[i] > r3 or close[i] > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals