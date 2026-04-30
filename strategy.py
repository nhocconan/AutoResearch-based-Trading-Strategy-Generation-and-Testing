#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop
# Donchian channels provide clear trend structure. Volume > 2x average confirms breakout strength.
# ATR(14) trailing stop (3x ATR) manages risk in both bull and bear markets.
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_Donchian20_VolumeBreakout_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for trailing stop
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    start_idx = max(20, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(high_rolling[i]) or 
            np.isnan(low_rolling[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian breakout
            if curr_volume_spike:
                # Bullish: price breaks above Donchian upper channel
                if curr_high > high_rolling[i]:
                    signals[i] = 0.25
                    position = 1
                    highest_high_since_entry = curr_high
                # Bearish: price breaks below Donchian lower channel
                elif curr_low < low_rolling[i]:
                    signals[i] = -0.25
                    position = -1
                    lowest_low_since_entry = curr_low
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_high_since_entry:
                highest_high_since_entry = curr_high
            
            # ATR trailing stop: exit if price drops 3x ATR from highest high
            if curr_close < (highest_high_since_entry - 3.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_low_since_entry:
                lowest_low_since_entry = curr_low
            
            # ATR trailing stop: exit if price rises 3x ATR from lowest low
            if curr_close > (lowest_low_since_entry + 3.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals