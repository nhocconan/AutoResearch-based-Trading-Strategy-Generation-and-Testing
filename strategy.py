#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_Volume
Hypothesis: Trade 4h Donchian(20) breakouts in direction of 12h EMA50 trend with volume confirmation.
Uses 4h primary timeframe for proven structure and 12h trend filter for robustness.
Donchian channels provide clear breakout levels; 12h EMA50 filters for higher timeframe trend alignment;
volume spike on 4h confirms breakout conviction. Works in bull/bear via trend filter + volume confirmation.
Target: 20-50 trades/year per symbol to avoid fee drag.
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
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels for today (based on prior 4h OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (prior 4h's levels available at 4h close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume confirmation: volume > 1.8x 20-period average on 4h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), Donchian (20), volume MA (20), aligned indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + price above 12h EMA50 + volume spike
            long_breakout = close[i] > donchian_high_aligned[i]
            long_signal = long_breakout and price_above_ema and volume_spike[i]
            
            # Short: price breaks below Donchian low + price below 12h EMA50 + volume spike
            short_breakout = close[i] < donchian_low_aligned[i]
            short_signal = short_breakout and price_below_ema and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches Donchian low OR trend turns bearish (price below EMA)
            if (close[i] < donchian_low_aligned[i] or not price_above_ema):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Donchian high OR trend turns bullish (price above EMA)
            if (close[i] > donchian_high_aligned[i] or not price_below_ema):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0