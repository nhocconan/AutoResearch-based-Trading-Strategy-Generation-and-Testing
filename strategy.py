#!/usr/bin/env python3
"""
6h_WeeklyDonchianBreakout_1dTrendFilter_VolumeSpike
Hypothesis: Trade weekly Donchian channel breakouts on 6h timeframe with 1d trend and volume filters.
- Weekly Donchian(20) from prior week provides major support/resistance levels
- 1d EMA50 ensures trades align with daily trend (works in bull/bear markets)
- Volume spike (2.5x 20-period average) confirms institutional participation
- Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
- Weekly structure reduces noise, trend/volume filters avoid false breakouts
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian High (20-period rolling max)
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Weekly Donchian Low (20-period rolling min)
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe (use previous week's levels to avoid look-ahead)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 20 for weekly Donchian/volume)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        
        if position == 0:
            # Long: breakout above weekly Donchian high AND close > 1d EMA50 AND volume spike
            if breakout_long and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below weekly Donchian low AND close < 1d EMA50 AND volume spike
            elif breakout_short and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price re-enters weekly Donchian channel (below midline)
            # Weekly midline = (donchian_high + donchian_low) / 2
            weekly_midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] < weekly_midline:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price re-enters weekly Donchian channel (above midline)
            weekly_midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] > weekly_midline:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchianBreakout_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0