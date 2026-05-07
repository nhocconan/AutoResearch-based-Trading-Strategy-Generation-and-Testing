#!/usr/bin/env python3
name = "6h_WeeklyTrend_Donchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channel (20-period) on 6h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_6h[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above 20-period high in weekly uptrend with volume
            if close[i] > high_20[i] and ema_50_6h[i] > ema_50_6h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-period low in weekly downtrend with volume
            elif close[i] < low_20[i] and ema_50_6h[i] < ema_50_6h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to 20-period low or weekly trend reverses
            if close[i] < low_20[i] or ema_50_6h[i] < ema_50_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to 20-period high or weekly trend reverses
            if close[i] > high_20[i] or ema_50_6h[i] > ema_50_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian breakout with weekly trend filter and volume confirmation
# - Donchian(20) breakout captures momentum and breakouts from consolidation
# - Weekly EMA50 trend filter ensures trades align with higher timeframe trend
# - Volume confirmation (2x average) reduces false breakouts
# - Exit on reversal of weekly trend or return to opposite Donchian band
# - Position size 0.25 balances return and risk (max 25% exposure)
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses weekly timeframe for trend and 6h for execution timing
# - Designed for low trade frequency (~15-25/year) to minimize fee drag
# - Novel combination: Donchian breakout + weekly trend + volume (not recently tested)