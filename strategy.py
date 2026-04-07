#!/usr/bin/env python3
"""
1h_ema_crossover_4h1d_volume_v1
Hypothesis: On 1-hour timeframe, use EMA crossover (fast 12, slow 26) for entry signals, 
filtered by 4h EMA trend direction and daily volume confirmation. Exit when price 
reverses across the fast EMA or volume confirmation fails. 
Designed for moderate frequency (15-37 trades/year) with trend-following edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_crossover_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate fast and slow EMA on 1h
    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA trend (21-period)
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 20-day average volume
    vol_20d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_20d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(26, n):  # Start after slow EMA warmup
        # Skip if 4h trend or daily volume not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(vol_20d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_20d_aligned[i]
        
        # EMA crossover signals
        ema_cross_up = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_down = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        if position == 1:  # Long position
            # Exit when EMA crosses down OR volume confirmation fails
            if ema_cross_down or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit when EMA crosses up OR volume confirmation fails
            if ema_cross_up or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: EMA crosses up, price above 4h EMA (uptrend), with volume confirmation
            long_entry = ema_cross_up and (close[i] > ema_4h_aligned[i]) and vol_confirm
            # Short entry: EMA crosses down, price below 4h EMA (downtrend), with volume confirmation
            short_entry = ema_cross_down and (close[i] < ema_4h_aligned[i]) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.20
            elif short_entry:
                position = -1
                signals[i] = -0.20
    
    return signals