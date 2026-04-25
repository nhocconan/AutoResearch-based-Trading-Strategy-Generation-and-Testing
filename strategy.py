#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA34 Trend + Volume Spike
Hypothesis: Daily Donchian breakouts capture medium-term trends, filtered by weekly EMA34 to ensure alignment with the primary trend. Volume spike confirms institutional participation. Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band). Daily timeframe targets 7-25 trades/year (30-100 over 4 years).
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period) on 1d data
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 20)  # Donchian channels, volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume_spike = volume_spike[i]
        
        # Trend filter: price relative to 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper AND uptrend AND volume spike
            long_entry = (curr_close > donchian_upper[i]) and uptrend and curr_volume_spike
            # Short: price breaks below Donchian lower AND downtrend AND volume spike
            short_entry = (curr_close < donchian_lower[i]) and downtrend and curr_volume_spike
            
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
            # Exit: price breaks below Donchian lower OR loss of uptrend
            if (curr_close < donchian_lower[i]) or (curr_close < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above Donchian upper OR loss of downtrend
            if (curr_close > donchian_upper[i]) or (curr_close > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0