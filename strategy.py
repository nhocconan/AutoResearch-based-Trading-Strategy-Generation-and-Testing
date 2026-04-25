#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Spike
Hypothesis: Donchian channel breakouts capture momentum. When aligned with 1d weekly pivot trend (bullish/bearish bias from weekly CPR) and confirmed by volume spikes, this filters false breakouts. Weekly pivot provides structural bias that works in both bull and bear regimes. Targets 12-35 trades/year by requiring confluence of three conditions.
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
    
    # Load 1d data ONCE before loop for weekly pivot trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Weekly pivot calculation using previous week's OHLC (simplified: use prior 5 days)
    # Weekly CPR (Central Pivot Range): (weekly high + weekly low + weekly close) / 3
    # We approximate weekly OHLC using rolling window of 5 days
    week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
    week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
    week_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly central pivot point
    weekly_pivot = (week_high + week_low + week_close) / 3.0
    
    # Bullish bias: price above weekly pivot, bearish bias: price below weekly pivot
    bullish_bias = close > weekly_pivot
    bearish_bias = close < weekly_pivot
    
    # Align weekly bias to 6h timeframe
    bullish_bias_aligned = align_htf_to_ltf(prices, df_1d, bullish_bias)
    bearish_bias_aligned = align_htf_to_ltf(prices, df_1d, bearish_bias)
    
    # Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) and weekly data (5-day lookback + 1 shift)
    start_idx = max(20, 5) + 1  # Donchian lookback + weekly lookback + shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(bullish_bias_aligned[i]) or np.isnan(bearish_bias_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Donchian breakout + weekly pivot bias + volume spike
            # Long: price breaks above Donchian upper band AND bullish bias AND volume spike
            long_entry = (curr_high > donchian_high[i]) and bullish_bias_aligned[i] and vol_spike
            # Short: price breaks below Donchian lower band AND bearish bias AND volume spike
            short_entry = (curr_low < donchian_low[i]) and bearish_bias_aligned[i] and vol_spike
            
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
            # Exit: price falls below Donchian lower band (mean reversion) OR loss of bullish bias
            if (curr_low < donchian_low[i]) or (~bullish_bias_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian upper band (mean reversion) OR loss of bearish bias
            if (curr_high > donchian_high[i]) or (~bearish_bias_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dWeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0