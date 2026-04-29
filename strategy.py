#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR-based stoploss
# Long when price breaks above 20-period Donchian high AND volume > 1.5x 20-bar avg
# Short when price breaks below 20-period Donchian low AND volume > 1.5x 20-bar avg
# Exit via ATR trailing stop: signal=0 when long position hits highest - 2.5*ATR or short hits lowest + 2.5*ATR
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 20-50 trades/year on 4h timeframe.
# Donchian channels provide structural breakouts, volume confirmation filters false breaks, ATR stop manages risk.

name = "4h_Donchian20_VolumeSpike_ATRStop_v1"
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
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(volume_ma_20[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_atr = atr[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        # Handle position management and exits
        if position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit when price drops 2.5*ATR from highest since entry
            if curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit when price rises 2.5*ATR from lowest since entry
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND volume confirmation
            if curr_close > curr_highest_high and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short when price breaks below Donchian low AND volume confirmation
            elif curr_close < curr_lowest_low and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals