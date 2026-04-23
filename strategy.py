#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation.
- Weekly pivot levels from prior 1d: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
- Trend filter: price > PP = bullish bias, price < PP = bearish bias
- Long: price breaks above Donchian(20) high + volume > 1.5x 20-period avg + price > weekly PP
- Short: price breaks below Donchian(20) low + volume > 1.5x 20-period avg + price < weekly PP
- Exit: price re-enters Donchian(20) channel OR weekly PP trend flip
- Uses Donchian for structure, weekly pivot for HTF bias, volume for conviction
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
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
    
    # Volume confirmation: > 1.5x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d weekly pivot points (based on prior 1d OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly pivot formula (using daily close as proxy for weekly)
    # PP = (high + low + close) / 3
    # R1 = 2*PP - low
    # S1 = 2*PP - high
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2.0 * pp - low_1d
    s1 = 2.0 * pp - high_1d
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(high_roll[i]) or
            np.isnan(low_roll[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume confirmation + price > weekly PP
            if (close[i] > high_roll[i] and 
                volume_confirm and 
                close[i] > pp_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume confirmation + price < weekly PP
            elif (close[i] < low_roll[i] and 
                  volume_confirm and 
                  close[i] < pp_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below Donchian low (mean reversion) OR price < weekly PP (trend flip)
            if close[i] < low_roll[i] or close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above Donchian high (mean reversion) OR price > weekly PP (trend flip)
            if close[i] > high_roll[i] or close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0