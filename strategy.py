#!/usr/bin/env python3
"""
12h Donchian(20) breakout + 1d EMA34 trend + volume confirmation
Hypothesis: Donchian breakouts capture momentum, EMA34 filters trend direction,
volume confirms breakout validity. Works in bull/bear via trend filter.
Target: 12-30 trades/year on 12h (50-150 total over 4 years).
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
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and EMA
    start_idx = max(40, 35)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Calculate Donchian channels (20-period) using only data up to i-1
        if i >= 20:
            # Use data from i-20 to i-1 (previous 20 completed bars)
            lookback_high = np.max(high[i-20:i])
            lookback_low = np.min(low[i-20:i])
        else:
            lookback_high = np.max(high[:i]) if i > 0 else curr_high
            lookback_low = np.min(low[:i]) if i > 0 else curr_low
        
        # Breakout conditions
        long_breakout = curr_close > lookback_high
        short_breakout = curr_close < lookback_low
        
        # Trend filter: price above/below EMA34
        above_ema = curr_close > ema_34_aligned[i]
        below_ema = curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: breakout + trend filter + volume spike
            long_entry = long_breakout and above_ema and vol_spike
            short_entry = short_breakout and below_ema and vol_spike
            
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
            # Exit: price breaks below Donchian low OR closes below EMA34
            if curr_close < lookback_low or curr_close < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above Donchian high OR closes above EMA34
            if curr_close > lookback_high or curr_close > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_EMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0