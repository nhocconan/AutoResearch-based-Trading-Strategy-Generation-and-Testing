#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeChopFilter_v1
Hypothesis: Trade Donchian(20) breakouts on 4h with volume confirmation and chop regime filter.
Donchian breakouts capture momentum in both bull and bear markets. Volume confirmation ensures
institutional participation. Chop regime filter (Bollinger Band Width percentile) avoids whipsaws
in ranging markets. Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF chop regime filter (BB Width percentile)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2) for chop regime
    close_1d = df_1d['close'].values
    bb_ma = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_ma
    
    # Calculate 50-period percentile rank of BB Width for chop regime
    # Chop = high BB Width (trending), Low BB Width (ranging)
    # We want to avoid ranging markets, so we look for LOW percentile (narrow BB = chop)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 50 else np.nan, raw=False
    ).values
    
    # Align chop regime to 4h timeframe (use previous day's value)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 4h volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), volume MA (20), chop regime (50)
    start_idx = max(20, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop regime filter: avoid ranging markets (low BB Width percentile = chop/choppy)
        # We want to trade when chop_regime_aligned[i] > 0.3 (avoid lowest 30% = ranging)
        # In ranging markets (low percentile), BB is narrow -> chop -> avoid
        # In trending markets (high percentile), BB is wide -> trend -> trade
        not_choppy = chop_regime_aligned[i] > 0.3
        
        if position == 0:
            # Long setup: price breaks above Donchian high + volume spike + not choppy
            long_setup = (close[i] > donchian_high[i]) and volume_spike[i] and not_choppy
            
            # Short setup: price breaks below Donchian low + volume spike + not choppy
            short_setup = (close[i] < donchian_low[i]) and volume_spike[i] and not_choppy
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Donchian low (stop) OR chop regime turns choppy (regime change)
            if (close[i] <= donchian_low[i]) or (not not_choppy):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian high (stop) OR chop regime turns choppy (regime change)
            if (close[i] >= donchian_high[i]) or (not not_choppy):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeChopFilter_v1"
timeframe = "4h"
leverage = 1.0