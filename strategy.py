#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation
# Donchian channels identify breakouts from price consolidation; EMA50 on 12h filters for higher timeframe trend alignment
# Volume spike (>2.0x 20-period average) confirms breakout strength; designed to work in both bull and bear markets
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe with discrete position sizing to minimize fee drag

name = "4h_Donchian20_EMA50_12h_VolumeSpike"
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
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian(20), 12h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian channels (20-period) up to current bar
        highest_high = np.max(high[i-19:i+1])  # 20-period high including current
        lowest_low = np.min(low[i-19:i+1])     # 20-period low including current
        
        # Calculate 20-period average volume for confirmation
        if i >= 19:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.nan
        
        if np.isnan(vol_ma_20):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * vol_ma_20
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR below 12h EMA50
            if curr_close < lowest_low or curr_close < curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR above 12h EMA50
            if curr_close > highest_high or curr_close > curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band + above 12h EMA50 + volume confirmation
            if (curr_close > highest_high and 
                curr_close > curr_ema_12h and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower band + below 12h EMA50 + volume confirmation
            elif (curr_close < lowest_low and 
                  curr_close < curr_ema_12h and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals