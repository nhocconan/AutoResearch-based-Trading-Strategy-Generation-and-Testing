#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Donchian Channel (20)
    highest_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Volume spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 2.0)
    
    # Align HTF indicators to 12h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 80  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_aligned[i]
        breakout_down = close[i] < lowest_low_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_aligned[i] > np.mean(atr_aligned[max(0, i-50):i+1]) * 0.5
        
        # Volume confirmation
        vol_confirm = vol_spike_aligned[i]
        
        # Entry conditions - Donchian breakout with volume and volatility
        long_entry = breakout_up and vol_confirm and vol_filter
        short_entry = breakout_down and vol_confirm and vol_filter
        
        # Exit conditions: opposite breakout or ATR-based trailing stop
        long_exit = breakout_down
        short_exit = breakout_up
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Breakout_Volume_Volatility"
timeframe = "12h"
leverage = 1.0