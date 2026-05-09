#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyCamarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (R1, S1, R3, S3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1 = pivot + (range_1w * 1.0 / 8)
    s1 = pivot - (range_1w * 1.0 / 8)
    
    # Align weekly levels to daily
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation (daily)
    vol_series = pd.Series(volume)
    vol_ma30 = vol_series.rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.3 * vol_ma30[i]
        
        if position == 0:
            # Long: Close above R1 with volume and above weekly EMA
            if (close[i] > r1_aligned[i]) and vol_ok and (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 with volume and below weekly EMA
            elif (close[i] < s1_aligned[i]) and vol_ok and (close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close below S1 or weekly trend reversal
            if (close[i] < s1_aligned[i]) or (close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close above R1 or weekly trend reversal
            if (close[i] > r1_aligned[i]) or (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals