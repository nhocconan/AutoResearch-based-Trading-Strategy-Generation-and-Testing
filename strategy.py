#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_ZeroCross_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # TRIX: Triple EMA of price, then rate of change
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (pd.Series(ema3).pct_change().values)  # TRIX as percentage
    
    # Get 1d volume for volume filter
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter (SMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align all to 4h
    trix_4h = align_htf_to_ltf(prices, df_1d, trix)
    vol_avg_1d_4h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    sma50_1w_4h = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Enough for warmup
    
    for i in range(start_idx, n):
        if (np.isnan(trix_4h[i]) or np.isnan(vol_avg_1d_4h[i]) or np.isnan(sma50_1w_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_4h[i]
        vol_avg = vol_avg_1d_4h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        trend = sma50_1w_4h[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume and above weekly SMA50
            if i > 0 and trix_4h[i-1] <= 0 and trix_val > 0 and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume and below weekly SMA50
            elif i > 0 and trix_4h[i-1] >= 0 and trix_val < 0 and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero or trend reversal
            if i > 0 and trix_4h[i-1] >= 0 and trix_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero or trend reversal
            if i > 0 and trix_4h[i-1] <= 0 and trix_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals