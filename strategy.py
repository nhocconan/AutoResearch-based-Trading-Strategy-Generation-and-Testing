#!/usr/bin/env python3
"""
6h_1d_1w_Weekly_Pivot_Trend_Follow
Hypothesis: Weekly pivot levels (from 1w) provide strong structural support/resistance.
Trend direction determined by daily EMA(50): price above = bullish, below = bearish.
Entry: 6h price breaks above weekly R1 in bullish mode or below weekly S1 in bearish mode,
with volume > 1.5x 20-period average. Exit on opposite signal.
Targets 15-30 trades/year per symbol. Works in bull/bear by following higher timeframe trend.
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivots (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift to use prior week's data (no look-ahead)
    high_prev = np.roll(high_1w, 1)
    low_prev = np.roll(low_1w, 1)
    close_prev = np.roll(close_1w, 1)
    high_prev[0] = high_1w[0]
    low_prev[0] = low_1w[0]
    close_prev[0] = close_1w[0]
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_1w = high_prev - low_prev
    
    # Weekly R1 and S1 (primary support/resistance)
    R1 = pivot + (range_1w * 1.0 / 2)
    S1 = pivot - (range_1w * 1.0 / 2)
    
    # Align weekly levels to 6h
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from daily EMA50
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        # Long entry: price breaks above weekly R1 in bullish trend with volume
        long_entry = bullish_trend and (close[i] > R1_aligned[i]) and volume_expansion[i]
        
        # Short entry: price breaks below weekly S1 in bearish trend with volume
        short_entry = bearish_trend and (close[i] < S1_aligned[i]) and volume_expansion[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_1w_Weekly_Pivot_Trend_Follow"
timeframe = "6h"
leverage = 1.0