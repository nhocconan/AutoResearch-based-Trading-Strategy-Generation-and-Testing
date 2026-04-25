#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with Weekly EMA Trend and Volume Spike
Hypothesis: Daily Donchian breakouts capture significant moves, filtered by weekly EMA trend and volume confirmation.
1d timeframe targets 7-25 trades/year, minimizing fee drag while capturing multi-week trends in both bull and bear markets.
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels (20-day) from previous daily bar
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    lookback = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_lower = low_series.rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(lookback, 20, 34) + 1  # Donchian + volume MA + EMA + 1 for shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Weekly trend filter: price above/below EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper AND uptrend AND volume spike
            long_entry = (curr_high > donchian_upper[i]) and uptrend and vol_spike
            # Short: price breaks below Donchian lower AND downtrend AND volume spike
            short_entry = (curr_low < donchian_lower[i]) and downtrend and vol_spike
            
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
            # Exit: price falls below Donchian lower OR loss of trend (price < weekly EMA34)
            if (curr_low < donchian_lower[i]) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian upper OR loss of trend (price > weekly EMA34)
            if (curr_high > donchian_upper[i]) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0