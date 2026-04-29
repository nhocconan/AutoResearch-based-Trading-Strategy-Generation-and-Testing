#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR trailing stop
# Uses Donchian channel from primary 4h timeframe for structure
# Volume confirmation (>1.8x 20-period average) ensures institutional participation
# ATR-based trailing stop (3.0 * ATR) manages risk and allows trends to run
# Designed for 4h timeframe to capture medium-term swings with controlled trade frequency (~20-50 trades/year)
# Works in both bull and bear markets by following price channels with volatility-adjusted stops

name = "4h_Donchian20_VolumeConfirm_ATRTrail_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 14)  # Donchian and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle position management and exits
        if position == 1:  # Long position
            # Trailing stop: exit if price drops 3.0 * ATR from highest since entry
            if curr_close < highest_since_entry - 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Update highest price since entry
                if curr_high > highest_since_entry:
                    highest_since_entry = curr_high
                    
        elif position == -1:  # Short position
            # Trailing stop: exit if price rises 3.0 * ATR from lowest since entry
            if curr_close > lowest_since_entry + 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Update lowest price since entry
                if curr_low < lowest_since_entry:
                    lowest_since_entry = curr_low
                    
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long entry: price breaks above upper Donchian channel with volume
            if vol_confirm and curr_close > high_roll[i]:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
            # Short entry: price breaks below lower Donchian channel with volume
            elif vol_confirm and curr_close < low_roll[i]:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals