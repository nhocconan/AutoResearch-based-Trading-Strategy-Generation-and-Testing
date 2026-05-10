#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: Camarilla R3/S3 levels from 12h act as breakout levels in 6h when aligned with 1d trend and volume spikes.
# Works in bull/bear markets by using 1d trend as filter and volume spikes to confirm breakouts.
# Targets 15-30 trades/year to minimize fee drag.

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R3, S3, R4, S4 calculation
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    camarilla_r3 = close_12h + (high_12h - low_12h) * 1.1 / 2
    camarilla_s3 = close_12h - (high_12h - low_12h) * 1.1 / 2
    camarilla_r4 = close_12h + (high_12h - low_12h) * 1.1
    camarilla_s4 = close_12h - (high_12h - low_12h) * 1.1
    
    # Align Camarilla levels to 6h
    r3_6h = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    r4_6h = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    s4_6h = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 6h
    trend_1d_up_6h = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_6h = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(trend_1d_up_6h[i]) or np.isnan(trend_1d_down_6h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R3 with 1d uptrend and volume spike
            if (high[i] > r3_6h[i] and 
                trend_1d_up_6h[i] > 0.5 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S3 with 1d downtrend and volume spike
            elif (low[i] < s3_6h[i] and 
                  trend_1d_down_6h[i] > 0.5 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S3 or reaches R4 (take profit)
            if (low[i] < s3_6h[i] or high[i] > r4_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R3 or reaches S4 (take profit)
            if (high[i] > r3_6h[i] or low[i] < s4_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals