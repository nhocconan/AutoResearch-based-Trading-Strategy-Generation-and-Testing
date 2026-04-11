#!/usr/bin/env python3
# 1d_1w_ema_trend_volume_v1
# Strategy: 1d EMA trend filter with weekly volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: EMA crossover on daily timeframe combined with weekly volume surge captures strong trends while avoiding chop. Weekly volume filter reduces false signals. Designed for low trade frequency (<20/year) to minimize fee drag in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA cross on daily timeframe (fast=21, slow=55)
    ema_fast = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_slow = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Calculate weekly volume average (20-period)
    vol_avg_20w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_20w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(55, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(vol_avg_20w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly volume confirmation: current week's volume > 1.8x 20-week average
        # Note: For daily data, we check if the current day belongs to a week with high volume
        vol_confirm = volume[i] > 1.8 * vol_avg_20w_aligned[i]
        
        # EMA crossover signals
        ema_cross_up = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_down = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # Entry conditions
        # Long: EMA bullish crossover AND volume confirmation
        if ema_cross_up and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: EMA bearish crossover AND volume confirmation
        elif ema_cross_down and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite EMA crossover
        elif position == 1 and ema_cross_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and ema_cross_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals