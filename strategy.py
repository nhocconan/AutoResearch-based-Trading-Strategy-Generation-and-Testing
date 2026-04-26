#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation
Hypothesis: On 6h timeframe, enter long when price breaks above 20-period Donchian high AND weekly pivot trend is up (price > weekly VWAP) AND volume > 1.8x 20-period average. Enter short when price breaks below 20-period Donchian low AND weekly pivot trend is down (price < weekly VWAP) AND volume spike. Uses Donchian channels for breakout structure, weekly VWAP for higher timeframe trend filter (works in bull/bear via mean reversion to VWAP), and volume confirmation for institutional participation. Targets 12-30 trades/year to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot approximation (using daily data to calculate weekly VWAP proxy)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly VWAP proxy: cumulative VWAP reset weekly
    # Since we don't have actual weekly data, use daily VWAP carried forward
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    volume_1d = df_1d['volume'].values
    vwap_1d = np.cumsum(typical_price_1d * volume_1d) / np.cumsum(volume_1d)
    vwap_1d = np.where(np.cumsum(volume_1d) == 0, typical_price_1d, vwap_1d)  # avoid div by zero
    
    # Align weekly VWAP proxy to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 20-period Donchian channels on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian warmup (20), volume MA warmup (20)
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Weekly pivot trend filter (using daily VWAP as proxy)
        trend_uptrend = close[i] > vwap_1d_aligned[i]
        trend_downtrend = close[i] < vwap_1d_aligned[i]
        
        if position == 0:
            # Long: price above Donchian high + weekly VWAP uptrend + volume spike
            long_signal = breakout_up and trend_uptrend and volume_spike[i]
            
            # Short: price below Donchian low + weekly VWAP downtrend + volume spike
            short_signal = breakout_down and trend_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below Donchian low OR trend change to downtrend
            if close[i] < lowest_low[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR trend change to uptrend
            if close[i] > highest_high[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0