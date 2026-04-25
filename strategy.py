#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with Weekly EMA34 Trend and Volume Spike
Hypothesis: Daily Donchian channel (20-day high/low) breakouts, aligned with weekly EMA34 trend
and confirmed by volume spikes, capture sustained momentum moves. The 1d timeframe targets
7-25 trades/year, minimizing fee drag while allowing proper trend alignment. Weekly EMA34
provides multi-timeframe trend filter that works in both bull and bear markets by ensuring
trades only occur in the direction of the higher timeframe trend.
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(34, 20)  # Weekly EMA34 + daily Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(high_ma[i]) or
            np.isnan(low_ma[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Weekly trend filter: price above/below weekly EMA34
        uptrend = ema_34_aligned[i] is not None and curr_close > ema_34_aligned[i]
        downtrend = ema_34_aligned[i] is not None and curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above 20-day high AND uptrend AND volume spike
            long_entry = (curr_high > high_ma[i]) and uptrend and vol_spike
            # Short: price breaks below 20-day low AND downtrend AND volume spike
            short_entry = (curr_low < low_ma[i]) and downtrend and vol_spike
            
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
            # Exit: price falls below 20-day low OR loss of trend (price < weekly EMA34)
            if (curr_low < low_ma[i]) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above 20-day high OR loss of trend (price > weekly EMA34)
            if (curr_high > high_ma[i]) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0