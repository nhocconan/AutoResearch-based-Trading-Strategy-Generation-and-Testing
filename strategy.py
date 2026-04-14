#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h pivot direction + volume confirmation
# Works in bull/bear: Donchian breakouts capture trending moves; 12h pivot acts as trend filter to avoid counter-trend entries
# Targets: 12-37 trades/year (50-150 over 4 years) by requiring 3-way confluence
# Position size: 0.25 to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # 12h pivot points from previous day
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate daily pivot from 12h data (using previous 12h bar's data)
    # Pivot = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    
    # Align pivot levels to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Donchian channels (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(pivot_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]):
            continue
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Long: Break above Donchian high + price above 12h R1 (bullish bias) + volume
        if position == 0 and high[i] > donchian_high[i] and close[i] > r1_12h_aligned[i] and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Break below Donchian low + price below 12h S1 (bearish bias) + volume
        elif position == 0 and low[i] < donchian_low[i] and close[i] < s1_12h_aligned[i] and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Opposite Donchian breakout or pivot reversal
        elif position != 0:
            if position == 1 and low[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and high[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian_12hPivot_Volume"
timeframe = "6h"
leverage = 1.0