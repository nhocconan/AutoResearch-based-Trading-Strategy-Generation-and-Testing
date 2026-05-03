#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above 20-period high with volume > 1.5x 20-period MA.
# Short when price breaks below 20-period low with volume > 1.5x 20-period MA.
# Uses ATR(14) for dynamic stoploss: exit long if price drops 2*ATR from entry high,
# exit short if price rises 2*ATR from entry low. Designed for 75-200 total trades over 4 years.
# Works in both bull and bear markets via symmetric breakout logic with volume filter.

name = "4h_Donchian20_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_high = 0.0  # highest high since long entry
    entry_low = 0.0   # lowest low since short entry
    
    for i in range(lookback, n):
        # Skip if any value is NaN
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_high = 0.0
                entry_low = 0.0
            continue
            
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Generate signals
        if position == 0:
            # Look for breakout entries with volume confirmation
            if close_val > highest_high[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_high = high_val
            elif close_val < lowest_low[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_low = low_val
        elif position == 1:
            # Update entry high for trailing stop
            entry_high = max(entry_high, high_val)
            # Exit long if price drops 2*ATR from entry high or breaks below Donchian low
            if close_val < entry_high - 2.0 * atr_val or close_val < lowest_low[i]:
                signals[i] = 0.0
                position = 0
                entry_high = 0.0
                entry_low = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update entry low for trailing stop
            entry_low = min(entry_low, low_val)
            # Exit short if price rises 2*ATR from entry low or breaks above Donchian high
            if close_val > entry_low + 2.0 * atr_val or close_val > highest_high[i]:
                signals[i] = 0.0
                position = 0
                entry_high = 0.0
                entry_low = 0.0
            else:
                signals[i] = -0.25
    
    return signals