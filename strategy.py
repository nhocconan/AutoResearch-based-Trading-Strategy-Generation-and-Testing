#!/usr/bin/env python3
"""
6h_WeeklyDonchian_1dTrend_VolumeBreakout
Hypothesis: 6h breakout of weekly Donchian channel (20-period) with 1d trend filter (price >/< EMA50) and volume confirmation (>2.0x 20-bar avg). 
Enters long when price breaks above weekly Donchian upper channel in 1d uptrend with volume spike, short when breaks below lower channel in 1d downtrend with volume spike. 
Exits on opposite Donchian breakout or trend reversal. 
Designed for 6h timeframe with ~10-25 trades/year, works in bull/bear by following 1d trend filter and capturing weekly momentum.
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
    
    # 1w data for HTF Donchian channel
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channel (20-period)
    donchian_upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 1 bar of previous data and warmup for indicators
    start_idx = max(20, 50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper_1w_aligned[i]) or 
            np.isnan(donchian_lower_1w_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper channel in 1d uptrend with volume confirmation
            long_setup = (close[i] > donchian_upper_1w_aligned[i]) and (close[i] > ema_50_1d_aligned[i]) and volume_spike[i]
            # Short: price breaks below weekly Donchian lower channel in 1d downtrend with volume confirmation
            short_setup = (close[i] < donchian_lower_1w_aligned[i]) and (close[i] < ema_50_1d_aligned[i]) and volume_spike[i]
            
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
            # Exit: price breaks below weekly Donchian lower channel OR trend turns down
            if (close[i] < donchian_lower_1w_aligned[i]) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above weekly Donchian upper channel OR trend turns up
            if (close[i] > donchian_upper_1w_aligned[i]) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchian_1dTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0