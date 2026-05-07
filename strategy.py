#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 12h data (using previous completed 12h bar)
    # Camarilla formula: based on previous day's range, but we'll use 12h bar
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Typical price for Camarilla calculation
    typical_12h = (h_12h + l_12h + c_12h) / 3
    range_12h = h_12h - l_12h
    
    # Camarilla levels (R3, S3 are most significant)
    r3 = typical_12h + range_12h * 1.1 / 2
    s3 = typical_12h - range_12h * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # 12h trend filter: EMA(50) on 12h close
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period average on 4h (approx 2.5 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 1.8
            uptrend = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            
            if close[i] > r3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and 12h downtrend
            elif close[i] < s3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below S3 or volume drops significantly
            if close[i] < s3_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above R3 or volume drops significantly
            if close[i] > r3_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h trend and volume confirmation
# - Camarilla R3/S3 levels act as strong support/resistance derived from 12h price action
# - Breakout above R3 with volume in 12h uptrend = high-probability long
# - Breakdown below S3 with volume in 12h downtrend = high-probability short
# - Volume confirmation (1.8x average) filters false breakouts
# - Trend filter ensures we trade with the 12h momentum
# - Works in bull markets (buy R3 breaks in uptrend) and bear markets (sell S3 breaks in downtrend)
# - Exit when price returns to S3/R3 or volume weakens
# - Position size 0.25 targets 20-35 trades/year, avoiding excessive fee drag
# - Uses actual 12h Camarilla levels (not resampled) via mtf_data for proper alignment
# - R3/S3 are the most significant Camarilla levels, reducing whipsaw vs R1/S1