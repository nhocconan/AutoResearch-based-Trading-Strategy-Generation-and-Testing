#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
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
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load daily data for Camarilla pivot levels (from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels (based on previous day)
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike detection: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above S3 with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            uptrend = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            
            if close[i] > S3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R3 with volume and 12h downtrend
            elif close[i] < R3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below S3 or volume drops
            if close[i] < S3_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above R3 or volume drops
            if close[i] > R3_aligned[i] or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h trend and volume confirmation
# - Camarilla R3/S3 levels act as strong support/resistance (based on prior day's range)
# - Breakout above S3 with volume in 12h uptrend = long opportunity
# - Breakdown below R3 with volume in 12h downtrend = short opportunity
# - Volume spike (2x average) confirms institutional participation
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to S3/R3 or volume weakens
# - Position size 0.25 targets 20-40 trades/year, avoiding fee drag
# - 12h trend filter ensures alignment with higher timeframe momentum
# - Uses actual Camarilla formula (not pivots) for precise levels