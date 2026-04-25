#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture momentum. Primary 12h timeframe reduces trade frequency to avoid fee drag. HTF 1d EMA34 filters trend direction (long only when price > EMA34, short when price < EMA34). Volume spike confirms participation. Works in bull (long on upside breakout) and bear (short on downside breakout). Target: 12-37 trades/year on 12h.
"""

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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period Donchian channels on primary 12h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-19:i+1])
        lowest_20[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA34, volume MA
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price > upper Donchian, above 1d EMA34, volume confirmation
            long_entry = (curr_close > upper_channel) and (curr_close > ema_34_val) and volume_confirm
            # Short: price < lower Donchian, below 1d EMA34, volume confirmation
            short_entry = (curr_close < lower_channel) and (curr_close < ema_34_val) and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below 1d EMA34 OR price < lower Donchian (stop and reverse)
            if curr_close < ema_34_val or curr_close < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above 1d EMA34 OR price > upper Donchian (stop and reverse)
            if curr_close > ema_34_val or curr_close > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0