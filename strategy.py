#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture momentum bursts. In strong trends (price > 1d EMA34),
breakouts above the upper band indicate bullish momentum; below lower band indicate bearish momentum.
Volume spike confirms institutional participation. Works in bull markets (buy breakouts in uptrend) 
and bear markets (sell breakdowns in downtrend). 12h timeframe targets 12-37 trades/year (50-150 over 4 years).
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) channels
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(donchian_window, 20, 34)  # Donchian, volume MA, 1d EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_upper = upper[i]
        curr_lower = lower[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Close > Upper Donchian AND uptrend AND volume spike
            long_entry = (curr_close > curr_upper) and uptrend and vol_spike
            # Short: Close < Lower Donchian AND downtrend AND volume spike
            short_entry = (curr_close < curr_lower) and downtrend and vol_spike
            
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
            # Exit: Close < Lower Donchian (breakdown) OR loss of uptrend
            if (curr_close < curr_lower) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Close > Upper Donchian (breakout) OR loss of downtrend
            if (curr_close > curr_upper) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0