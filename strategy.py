#!/usr/bin/env python3
name = "6h_12h_Financial_Star_Trend_Volume"
timeframe = "6h"
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
    
    # Calculate 12h Financial Star (modified Williams %R)
    # Financial Star = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 = overbought, -100 = oversold
    period = 14
    highest_high = pd.Series(df_12h['high']).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=period, min_periods=period).min().values
    fs_raw = ((highest_high - df_12h['close'].values) / (highest_high - lowest_low)) * -100
    fs_raw = np.where((highest_high - lowest_low) == 0, -50, fs_raw)  # avoid div by zero
    
    # Align Financial Star to 6h
    fs_aligned = align_htf_to_ltf(prices, df_12h, fs_raw)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection: 3-period average (1.5 days of 6h bars)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 3)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(fs_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Financial Star oversold (< -80) with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_3[i] * 2.0
            uptrend = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            
            if fs_aligned[i] < -80 and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Financial Star overbought (> -20) with volume and 12h downtrend
            elif fs_aligned[i] > -20 and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Financial Star returns to neutral (-50) or volume drops
            if fs_aligned[i] > -50 or volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Financial Star returns to neutral (-50) or volume drops
            if fs_aligned[i] < -50 or volume[i] < vol_ma_3[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Financial Star (Williams %R variant) with 12h trend and volume confirmation
# - Financial Star identifies overbought/oversold conditions on 12h timeframe
# - Long when FS < -80 (deep oversold) with volume spike in 12h uptrend
# - Short when FS > -20 (deep overbought) with volume spike in 12h downtrend
# - Volume spike (2.0x average) confirms institutional participation at extremes
# - Exit when FS returns to neutral (-50) or volume weakens
# - Works in both bull (buy oversold in uptrend) and bear (sell overbought in downtrend)
# - Position size 0.25 targets ~15-40 trades/year, avoiding fee drag
# - Uses 12h Financial Star as novel oscillator (not commonly tried)
# - 12h trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false signals at extremes
# - Novel combination: Financial Star (12h) + trend (12h) + volume (6h) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits