#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and EMA200
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # CP = (H + L + C) / 3
    # Range = H - L
    # R3 = CP + Range * 1.1 / 2
    # S3 = CP - Range * 1.1 / 2
    # R4 = CP + Range * 1.1
    # S4 = CP - Range * 1.1
    cp = (high_1d + low_1d + close_1d) / 3.0
    rang = high_1d - low_1d
    r3 = cp + rang * 1.1 / 2.0
    s3 = cp - rang * 1.1 / 2.0
    r4 = cp + rang * 1.1
    s4 = cp - rang * 1.1
    
    # EMA200 on daily close
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    ema_200_1d_6h = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Trend filter: price vs EMA200 (1d)
    uptrend = close > ema_200_1d_6h
    downtrend = close < ema_200_1d_6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_200_1d_6h[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 or trend reverses
            if close[i] < s3_6h[i] or not uptrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 or trend reverses
            if close[i] > r3_6h[i] or not downtrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at S3/R3 in ranging market (price near S3/R3 + volume spike)
            # Breakout continuation at S4/R4 in trending market
            near_s3 = abs(close[i] - s3_6h[i]) < (r3_6h[i] - s3_6h[i]) * 0.05  # within 5% of S3-R3 range
            near_r3 = abs(close[i] - r3_6h[i]) < (r3_6h[i] - s3_6h[i]) * 0.05  # within 5% of S3-R3 range
            
            # Long: price near S3 + volume spike + uptrend (breakout continuation)
            if near_s3 and vol_spike[i] and uptrend[i] and close[i] > s4_6h[i]:
                position = 1
                signals[i] = 0.25
            # Short: price near R3 + volume spike + downtrend (breakout continuation)
            elif near_r3 and vol_spike[i] and downtrend[i] and close[i] < r4_6h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals