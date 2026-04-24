#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for weekly pivot levels (R1/S1, R2/S2).
- Weekly pivot levels act as strong support/resistance: price tends to respect these levels.
- Entry: Long when price breaks above Donchian(20) upper AND close > weekly R1 (bullish bias).
         Short when price breaks below Donchian(20) lower AND close < weekly S1 (bearish bias).
         Weekly pivot acts as regime filter: only take breakouts aligned with weekly bias.
- Exit: Opposite Donchian breakout or price crosses weekly pivot (R1/S1) in opposite direction.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull and bear markets: weekly pivot adapts to longer-term trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Shift by 1 to use prior week's data (no look-ahead)
    weekly_high_shift = np.roll(weekly_high, 1)
    weekly_low_shift = np.roll(weekly_low, 1)
    weekly_close_shift = np.roll(weekly_close, 1)
    # First value will be invalid (rolled from last), set to NaN
    weekly_high_shift[0] = np.nan
    weekly_low_shift[0] = np.nan
    weekly_close_shift[0] = np.nan
    
    # Calculate weekly pivot levels
    weekly_pivot = (weekly_high_shift + weekly_low_shift + weekly_close_shift) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low_shift
    weekly_s1 = 2 * weekly_pivot - weekly_high_shift
    weekly_r2 = weekly_pivot + (weekly_high_shift - weekly_low_shift)
    weekly_s2 = weekly_pivot - (weekly_high_shift - weekly_low_shift)
    
    # Align weekly pivot levels to 6h
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Donchian channels (20-period) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)  # Need enough bars for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                # Bullish breakout: price closes above upper Donchian AND above weekly R1 (bullish bias)
                if curr_close > highest_high[i] and curr_close > weekly_r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below lower Donchian AND below weekly S1 (bearish bias)
                elif curr_close < lowest_low[i] and curr_close < weekly_s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below Donchian mid OR crosses below weekly S1 (bearish shift)
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if curr_close < donchian_mid or curr_close < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR crosses above weekly R1 (bullish shift)
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if curr_close > donchian_mid or curr_close > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wPivotRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0