#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR stoploss
# - Uses 12h Donchian channel breakouts for trend following entries
# - Confirmed by 1d volume spike (>1.5x 20-period average volume)
# - ATR-based trailing stop (2.0x ATR) to manage risk
# - Fixed position size 0.25 to control drawdown
# - Works in both bull and bear markets by capturing medium-term trends
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_donchian_breakout_volume_atr_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d average volume (20-period)
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d average volume to 12h timeframe
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Pre-compute ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume
        # Note: we use the 1d volume that corresponds to the completed 1d bar
        volume_spike = volume_1d[-1] > 1.5 * avg_volume_1d_aligned[i] if len(volume_1d) > 0 and not np.isnan(avg_volume_1d_aligned[i]) else False
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.0x ATR from highest high
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.0x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above Donchian upper band with volume confirmation
            if close[i] > highest_high[i] and volume_spike:
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band with volume confirmation
            elif close[i] < lowest_low[i] and volume_spike:
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals