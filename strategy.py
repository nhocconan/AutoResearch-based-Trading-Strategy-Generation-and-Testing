#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Trend_Volume_V2
Strategy: 4h Donchian channel breakout with 1d trend filter and volume confirmation.
Long: Price breaks above Donchian(20) high + 1d close > 1d SMA(50) + volume > 1.5x avg.
Short: Price breaks below Donchian(20) low + 1d close < 1d SMA(50) + volume > 1.5x avg.
Exit: Opposite Donchian break or trend reversal.
Target: 20-40 trades/year per symbol (80-160 total over 4 years).
Works in bull via trend-following, in bear via short side.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily SMA(50) for trend filter
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Donchian channels (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for Donchian and SMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(sma_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        trend_up = close_1d[i] > sma_50_1d[i]  # daily trend
        trend_down = close_1d[i] < sma_50_1d[i]
        
        if position == 0:
            # Long: Donchian breakout up + uptrend + volume
            if close[i] > donch_high[i] and trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + downtrend + volume
            elif close[i] < donch_low[i] and trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Donchian breakdown OR trend reversal
            if close[i] < donch_low[i] or not trend_up:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Donchian breakout OR trend reversal
            if close[i] > donch_high[i] or not trend_down:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Trend_Volume_V2"
timeframe = "4h"
leverage = 1.0