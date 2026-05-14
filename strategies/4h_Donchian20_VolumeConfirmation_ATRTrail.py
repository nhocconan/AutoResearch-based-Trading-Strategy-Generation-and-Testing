#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and ATR-based trailing stop
# Uses Donchian(20) on 4h timeframe to capture institutional breakouts in both bull and bear markets
# Volume confirmation (>2.5x 20-period average) filters false breakouts
# ATR(14) trailing stop (3x ATR) manages risk without whipsaw
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year) minimizing fee drag
# BTC/ETH focus: requires volume confirmation to avoid SOL-only bias

name = "4h_Donchian20_VolumeConfirmation_ATRTrail"
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
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate upper and lower Donchian channels
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (delayed by one 4h bar)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Calculate ATR(14) for trailing stop
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
    
    start_idx = max(20, 20, 14)  # Donchian, volume MA, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_20_aligned[i]
        curr_lower = lower_20_aligned[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle trailing stop and exits
        if position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Trailing stop: price drops 3*ATR from highest since entry
            if curr_close < highest_since_entry - 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below lower Donchian channel
            elif curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Trailing stop: price rises 3*ATR from lowest since entry
            if curr_close > lowest_since_entry + 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above upper Donchian channel
            elif curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.5x 20-period average
            vol_confirm = curr_volume > 2.5 * curr_vol_ma
            
            # Long entry: price breaks above upper Donchian with volume confirmation
            if vol_confirm and curr_high > curr_upper:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short entry: price breaks below lower Donchian with volume confirmation
            elif vol_confirm and curr_low < curr_lower:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals