#!/usr/bin/env python3
# 12h_1w_1d_ema_crossover_volume_v1
# Hypothesis: Trade EMA crossovers on 12h timeframe with weekly trend filter and daily volume confirmation.
# Long when 12h EMA21 crosses above EMA50 with weekly uptrend (price > weekly EMA50) and daily volume surge.
# Short when 12h EMA21 crosses below EMA50 with weekly downtrend (price < weekly EMA50) and daily volume surge.
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Weekly trend filter ensures alignment with higher timeframe momentum, working in both bull and bear markets.
# Volume surge confirms breakout strength and reduces false signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_ema_crossover_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA21 and EMA50 for 12h
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA50 for weekly trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-day average volume for daily volume confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly EMA50 and daily volume MA to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21[i]) or np.isnan(ema50[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition: volume > 1.8x 20-day average
        vol_surge = volume[i] > 1.8 * vol_ma_20_1d_aligned[i] if vol_ma_20_1d_aligned[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: EMA21 crosses below EMA50
            if ema21[i] < ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA21 crosses above EMA50
            if ema21[i] > ema50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA21 crosses above EMA50 with weekly uptrend and volume surge
            if (ema21[i] > ema50[i] and ema21[i-1] <= ema50[i-1] and  # crossover
                close[i] > ema50_1w_aligned[i] and vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: EMA21 crosses below EMA50 with weekly downtrend and volume surge
            elif (ema21[i] < ema50[i] and ema21[i-1] >= ema50[i-1] and  # crossunder
                  close[i] < ema50_1w_aligned[i] and vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals