#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with weekly Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation.
Long when price breaks above weekly Donchian upper channel with volume confirmation and price > 1d EMA50 (uptrend).
Short when price breaks below weekly Donchian lower channel with volume confirmation and price < 1d EMA50 (downtrend).
Exit when price returns to the weekly Donchian midpoint (mean reversion to channel center).
Uses weekly timeframe for structure (reduces noise) and 6h for entry timing and trend filter.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
Weekly Donchian provides strong support/resistance levels that work in both bull and bear markets.
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
    
    # Get weekly data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian(20) channels
    lookback = 20
    upper_1w = pd.Series(high_1w).rolling(window=lookback, min_periods=lookback).max().values
    lower_1w = pd.Series(low_1w).rolling(window=lookback, min_periods=lookback).min().values
    mid_1w = (upper_1w + lower_1w) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly Donchian levels to 6h timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    mid_1w_aligned = align_htf_to_ltf(prices, df_1w, mid_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_1w_aligned[i]) or 
            np.isnan(lower_1w_aligned[i]) or 
            np.isnan(mid_1w_aligned[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper with volume and uptrend (price > EMA50)
            if (close[i] > upper_1w_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower with volume and downtrend (price < EMA50)
            elif (close[i] < lower_1w_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below weekly Donchian midpoint
            if close[i] <= mid_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above weekly Donchian midpoint
            if close[i] >= mid_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wDonchian20_Breakout_Volume_EMA50_Trend"
timeframe = "6h"
leverage = 1.0