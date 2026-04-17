#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Regime
Hypothesis: Donchian(20) breakout with volume confirmation and choppiness regime filter.
Long when price breaks above 20-period high + volume > 1.5x avg + CHOP > 61.8 (range).
Short when price breaks below 20-period low + volume > 1.5x avg + CHOP > 61.8.
Exit on opposite breakout. Position size: ±0.25. Works in bull (trend) and bear (range reversion).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def donchian_channel(high, low, window=20):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Volume confirmation (10-period MA)
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Choppiness Index (14-period)
    def choppiness_index(high, low, close, window=14):
        atr = []
        for i in range(len(close)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr.append(tr)
        atr = np.array(atr)
        
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        chop = np.zeros_like(close)
        for i in range(window-1, len(close)):
            if highest_high[i] - lowest_low[i] > 0:
                chop[i] = 100 * np.log10(sum_atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(window)
            else:
                chop[i] = 50
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # Get 1D data for additional regime filter (optional trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1D EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 10, 14, 50)  # Donchian, volume MA, chop, EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma10[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 10-period average
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        # Regime filter: choppy market (CHOP > 61.8) for mean reversion
        regime_filter = chop[i] > 61.8
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Break above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # Break below previous lower band
        
        if position == 0:
            # Long: Donchian breakout up + volume filter + choppy regime
            if breakout_up and volume_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volume filter + choppy regime
            elif breakout_down and volume_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian breakout down
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian breakout up
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0