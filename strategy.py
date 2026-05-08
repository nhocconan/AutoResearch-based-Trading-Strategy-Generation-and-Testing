#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d ATR(14) for volatility normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])
    
    atr14 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Calculate daily Camarilla levels (based on previous day)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    range_prev = high_prev - low_prev
    camarilla_r1 = close_prev + 1.1 * range_prev / 12
    camarilla_s1 = close_prev - 1.1 * range_prev / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike detection: current volume > 2 * 20-period average (12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(atr14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_1d_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R1 with volume spike, above 1d EMA
            if (close[i] > r1 and vol_spike and close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with volume spike, below 1d EMA
            elif (close[i] < s1 and vol_spike and close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 OR below 1d EMA
            if (close[i] < s1 or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 OR above 1d EMA
            if (close[i] > r1 or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals