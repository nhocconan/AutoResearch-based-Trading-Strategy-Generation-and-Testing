#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_Combined"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 1d close for pivot calculation (previous day)
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]  # first value
    
    # Calculate pivot and ranges
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    pivot_1d = (high_1d + low_1d + close_1d_prev) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3, S3 (most significant)
    r3_1d = close_1d_prev + range_1d * 1.1 / 2
    s3_1d = close_1d_prev - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR filter: only trade when ATR is between 0.5% and 5% of price
    atr_percent = atr14_1d_aligned / close
    atr_filter = (atr_percent >= 0.005) & (atr_percent <= 0.05)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for daily calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(atr_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_1d_aligned[i]
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_ok = atr_filter[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike, above 1d EMA, in ATR range
            if (close[i] > r3_val and vol_spike and 
                close[i] > ema_val and atr_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike, below 1d EMA, in ATR range
            elif (close[i] < s3_val and vol_spike and 
                  close[i] < ema_val and atr_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR below 1d EMA
            if (close[i] < s3_val or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR above 1d EMA
            if (close[i] > r3_val or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals