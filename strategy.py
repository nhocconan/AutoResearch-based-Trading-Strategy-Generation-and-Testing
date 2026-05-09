#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
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
    
    # Get 1w data for trend (HMA21)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w HMA21 for trend
    close_1w = df_1w['close'].values
    half_length = int(21 / 2)
    sqrt_length = int(np.sqrt(21))
    wma1 = pd.Series(close_1w).ewm(span=half_length, adjust=False).mean()
    wma2 = pd.Series(close_1w).ewm(span=21, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma_21w = raw_hma.ewm(span=sqrt_length, adjust=False).mean().values
    
    # 1d Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    camarilla_high = close_1d + 1.1 * range_1d / 12  # R3 level
    camarilla_low = close_1d - 1.1 * range_1d / 12   # S3 level
    
    # 1d volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    hma_21w_12h = align_htf_to_ltf(prices, df_1w, hma_21w)
    camarilla_high_12h = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_12h = align_htf_to_ltf(prices, df_1d, camarilla_low)
    vol_avg_1d_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(hma_21w_12h[i]) or np.isnan(camarilla_high_12h[i]) or 
            np.isnan(camarilla_low_12h[i]) or np.isnan(vol_avg_1d_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = hma_21w_12h[i]
        resistance = camarilla_high_12h[i]
        support = camarilla_low_12h[i]
        vol_avg = vol_avg_1d_12h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        if position == 0:
            # Long: break above R3 with volume and above weekly HMA21
            if close[i] > resistance and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and below weekly HMA21
            elif close[i] < support and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S3 or trend reversal
            if close[i] < support or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R3 or trend reversal
            if close[i] > resistance or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals