#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss
# Uses Donchian channel breakouts for structural entry signals
# Volume confirmation > 1.5x average to filter weak breakouts
# ATR-based trailing stoploss to manage risk in both bull and bear markets
# Discrete position sizing (0.25) to minimize fee churn
# Designed for fewer, higher-quality trades with strong risk control

name = "4h_Donchian20_VolumeSpike_ATRStop_v2"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Donchian channel (20-period)
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate ATR (14-period) for stoploss
    atr_period = 14
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(donchian_window, atr_period, 20)  # Warmup period
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle position management and exits
        if position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            
            # Exit conditions:
            # 1. ATR-based trailing stop: price drops 2.5*ATR from highest since entry
            # 2. Mean reversion: price retreats to middle of Donchian channel
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2.0
            if curr_close <= highest_since_entry - 2.5 * curr_atr or curr_close <= donchian_middle:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Exit conditions:
            # 1. ATR-based trailing stop: price rises 2.5*ATR from lowest since entry
            # 2. Mean reversion: price advances to middle of Donchian channel
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2.0
            if curr_close >= lowest_since_entry + 2.5 * curr_atr or curr_close >= donchian_middle:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above upper Donchian band with volume confirmation
            if curr_high > highest_high[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
            # Short when price breaks below lower Donchian band with volume confirmation
            elif curr_low < lowest_low[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals