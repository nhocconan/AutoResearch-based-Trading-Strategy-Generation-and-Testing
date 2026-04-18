#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
- Long when price breaks above Donchian high (20) AND weekly pivot shows bullish bias (price > weekly pivot)
- Short when price breaks below Donchian low (20) AND weekly pivot shows bearish bias (price < weekly pivot)
- Weekly pivot provides structural bias from higher timeframe to avoid counter-trend trades
- Volume confirmation ensures breakouts have conviction
- Designed for 15-25 trades/year to minimize fee drag while capturing strong trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper band = highest high, lower band = lowest low."""
    upper = np.full(len(high), np.nan)
    lower = np.full(len(low), np.nan)
    
    for i in range(period - 1, len(high)):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P = (H + L + C)/3."""
    pivot = (high + low + close) / 3.0
    return pivot

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot points on weekly data
    pivot_1w = calculate_pivot_points(high_1w, low_1w, close_1w)
    
    # Calculate Donchian channels on 6h data (20-period)
    donch_high, donch_low = calculate_donchian_channels(high, low, 20)
    
    # Align weekly pivot to 6h timeframe
    pivot_1w_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(pivot_1w_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, above weekly pivot, volume confirmation
            if close[i] > donch_high[i] and close[i] > pivot_1w_6h[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below weekly pivot, volume confirmation
            elif close[i] < donch_low[i] and close[i] < pivot_1w_6h[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below Donchian high or below weekly pivot
            if close[i] <= donch_high[i] or close[i] <= pivot_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above Donchian low or above weekly pivot
            if close[i] >= donch_low[i] or close[i] >= pivot_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0