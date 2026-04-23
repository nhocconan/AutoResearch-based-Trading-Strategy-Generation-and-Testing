#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Width Regime + Donchian Breakout with Weekly Pivot Filter.
- Bollinger Band Width (BBW) identifies regime: low BBW = squeeze (range), high BBW = expansion (trend)
- In range regime (BBW < 30th percentile): mean reversion at Donchian(10) channels
- In trend regime (BBW > 70th percentile): breakout continuation at Donchian(20) channels
- Weekly pivot (from 1w data) as directional filter: only long above weekly pivot, short below
- Volume confirmation: breakout/mean reversion signals require volume > 1.3x average
- Position size: 0.25 discrete level
- Works in bull/bear via regime adaptation and weekly pivot filter
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
    
    # Bollinger Bands (20, 2) for Band Width
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = basis + 2.0 * dev
    lower_bb = basis - 2.0 * dev
    bb_width = (upper_bb - lower_bb) / basis  # Normalized BB Width
    
    # Percentile lookback for regime (50 periods ~ 6h*50 = ~12.5 days)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=50, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Donchian channels
    donch_10_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donch_10_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donch_20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # BBands, BBW percentile, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(basis[i]) or np.isnan(dev[i]) or np.isnan(bb_width[i]) or
            np.isnan(bb_width_pct[i]) or np.isnan(donch_10_high[i]) or np.isnan(donch_10_low[i]) or
            np.isnan(donch_20_high[i]) or np.isnan(donch_20_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Regime detection
        is_range = bb_width_pct[i] < 0.30   # Low BBW = squeeze/range
        is_trend = bb_width_pct[i] > 0.70   # High BBW = expansion/trend
        
        if position == 0:
            # Long conditions
            long_signal = False
            if is_range and close[i] <= donch_10_low[i] and volume_confirm:
                # Mean reversion long at lower Donchian(10) in range
                long_signal = True
            elif is_trend and close[i] >= donch_20_high[i] and volume_confirm:
                # Breakout long at upper Donchian(20) in trend
                long_signal = True
            
            # Short conditions
            short_signal = False
            if is_range and close[i] >= donch_10_high[i] and volume_confirm:
                # Mean reversion short at upper Donchian(10) in range
                short_signal = True
            elif is_trend and close[i] <= donch_20_low[i] and volume_confirm:
                # Breakout short at lower Donchian(20) in trend
                short_signal = True
            
            # Apply weekly pivot filter
            if long_signal and close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif short_signal and close[i] < weekly_pivot_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses Donchian(10) mean OR regime shifts strongly
            exit_signal = False
            if is_range and close[i] >= (donch_10_high[i] + donch_10_low[i]) / 2.0:
                exit_signal = True  # Exit mean reversion at midpoint
            elif is_trend and close[i] <= donch_20_low[i]:
                exit_signal = True  # Exit trend breakout if price fails
            elif bb_width_pct[i] > 0.85:  # Extreme expansion may precede reversal
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses Donchian(10) mean OR regime shifts strongly
            exit_signal = False
            if is_range and close[i] <= (donch_10_high[i] + donch_10_low[i]) / 2.0:
                exit_signal = True  # Exit mean reversion at midpoint
            elif is_trend and close[i] >= donch_20_high[i]:
                exit_signal = True  # Exit trend breakout if price fails
            elif bb_width_pct[i] > 0.85:  # Extreme expansion may precede reversal
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BBWRegime_DonchianBreakout_WeeklyPivot_v1"
timeframe = "6h"
leverage = 1.0