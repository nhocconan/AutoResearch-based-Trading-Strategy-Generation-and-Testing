#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture momentum. Primary timeframe 12h reduces trade frequency. 1d EMA34 establishes trend filter. Volume spike confirms participation. Works in bull (long on upside breakout above EMA34) and bear (short on downside breakout below EMA34). Target: 12-37 trades/year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20), EMA34, volume MA
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if EMA34 not ready
        if np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian(20) on primary timeframe using available data
        lookback = min(20, i + 1)  # Use available data up to current bar
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        
        # Calculate 20-period volume MA for volume confirmation
        vol_lookback = min(20, i + 1)
        vol_ma = np.mean(volume[i-vol_lookback+1:i+1]) if vol_lookback > 0 else 0
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price > Donchian upper, above 1d EMA34, volume confirmation
            long_entry = (curr_close > highest_high) and (curr_close > ema_34_val) and volume_confirm
            # Short: price < Donchian lower, below 1d EMA34, volume confirmation
            short_entry = (curr_close < lowest_low) and (curr_close < ema_34_val) and volume_confirm
            
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
            # Exit: price crosses below Donchian lower OR 1d EMA34 (stop and reverse)
            if curr_close < lowest_low or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Donchian upper OR 1d EMA34 (stop and reverse)
            if curr_close > highest_high or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0