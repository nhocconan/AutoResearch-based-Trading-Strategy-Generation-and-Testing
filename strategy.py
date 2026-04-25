#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Trade 12h timeframe using Donchian(20) breakout for entry, 
weekly EMA34 for trend filter, and daily volume spike (>2.0x 20-bar MA) for confirmation. 
Enter long when price breaks above Donchian(20) high AND above weekly EMA34 AND volume spike. 
Enter short when price breaks below Donchian(20) low AND below weekly EMA34 AND volume spike. 
Exit on opposite Donchian touch or trend reversal. Uses discrete sizing 0.25 to balance 
return and drawdown. Target 12-37 trades/year on 12h timeframe. Works in bull/bear via 
weekly trend filter and volatility-based breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for daily volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), weekly EMA (34), daily volume MA (20)
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian channels for 20-bar lookback (using only past data)
        highest_high = np.max(high[i-19:i+1]) if i >= 19 else np.nan
        lowest_low = np.min(low[i-19:i+1]) if i >= 19 else np.nan
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly EMA34 AND volume spike
            long_setup = (not np.isnan(highest_high)) and \
                         (close[i] > highest_high) and \
                         (close[i] > ema_34_1w_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price breaks below Donchian low AND below weekly EMA34 AND volume spike
            short_setup = (not np.isnan(lowest_low)) and \
                          (close[i] < lowest_low) and \
                          (close[i] < ema_34_1w_aligned[i]) and \
                          volume_spike_1d_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Donchian low OR closes below weekly EMA34
            if (not np.isnan(lowest_low)) and \
               (close[i] <= lowest_low) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian high OR closes above weekly EMA34
            if (not np.isnan(highest_high)) and \
               (close[i] >= highest_high) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0