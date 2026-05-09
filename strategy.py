#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily high, low, close for Camarilla
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_R3 = np.zeros(len(df_1d))
    camarilla_S3 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i >= 0:  # Need at least one day
            H = high_1d[i]
            L = low_1d[i]
            C = close_1d[i]
            camarilla_R3[i] = C + (H - L) * 1.1 / 6
            camarilla_S3[i] = C - (H - L) * 1.1 / 6
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume spike (20-period average)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    # Align Camarilla levels and trend to 12h
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = volume[i] > 1.5 * vol_avg_20[i]
        
        price = close[i]
        r3 = camarilla_R3_aligned[i]
        s3 = camarilla_S3_aligned[i]
        ema34_today = ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume and above daily EMA34
            if price > r3 and vol_confirmed and price > ema34_today:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and below daily EMA34
            elif price < s3 and vol_confirmed and price < ema34_today:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or trend changes
            if price < s3 or price < ema34_today:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 or trend changes
            if price > r3 or price > ema34_today:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals