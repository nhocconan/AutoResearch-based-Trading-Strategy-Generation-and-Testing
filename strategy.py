#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_elder_ray_momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Elder Ray calculation (13-period EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    ema13 = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1w - ema13
    bear_power = low_1w - ema13
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    # 6h EMA13 for trend filter
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema13_6h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: Bull Power > 0 (strong buying pressure) + price above EMA13 + volume
        long_signal = (bull_power_aligned[i] > 0 and 
                      close[i] > ema13_6h[i] and 
                      volume_filter[i])
        
        # Short: Bear Power < 0 (strong selling pressure) + price below EMA13 + volume
        short_signal = (bear_power_aligned[i] < 0 and 
                       close[i] < ema13_6h[i] and 
                       volume_filter[i])
        
        # Exit: Elder Ray divergence or loss of momentum
        exit_long = (position == 1 and 
                    (bull_power_aligned[i] <= 0 or close[i] < ema13_6h[i]))
        exit_short = (position == -1 and 
                     (bear_power_aligned[i] >= 0 or close[i] > ema13_6h[i]))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals