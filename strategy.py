#!/usr/bin/env python3
"""
12h Donchian(20) breakout + 1d EMA50 trend + volume spike
Hypothesis: Donchian breakouts capture momentum bursts. 1d EMA50 provides higher-timeframe trend filter to avoid counter-trend trades. Volume spike confirms institutional participation. Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend). 12h timeframe targets 12-37 trades/year (50-150 over 4 years).
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
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50)  # Donchian period, 1d EMA50 alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_aligned[i]
        downtrend = curr_close < ema_50_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND uptrend AND volume spike
            long_entry = (curr_high > donchian_high[i-1]) and uptrend and vol_spike
            # Short: price breaks below Donchian low AND downtrend AND volume spike
            short_entry = (curr_low < donchian_low[i-1]) and downtrend and vol_spike
            
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
            # Exit: price breaks below Donchian low OR loss of uptrend
            if (curr_low < donchian_low[i]) or (curr_close < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above Donchian high OR loss of downtrend
            if (curr_high > donchian_high[i]) or (curr_close > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0