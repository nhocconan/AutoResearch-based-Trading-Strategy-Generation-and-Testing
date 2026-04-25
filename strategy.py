#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrendFilter_VolumeConfirm
Hypothesis: Donchian(20) breakouts on 1d with 1w EMA50 trend filter and volume confirmation. 
In trending markets (price > EMA50 on 1w for longs, price < EMA50 on 1w for shorts), 
breakouts in trend direction have higher success. Volume confirms breakout validity. 
Uses discrete position sizing (0.25) to minimize fee churn. 
Target: 15-30 trades/year, works in both bull/bear by following the 1w trend.
Timeframe: 1d, leverage: 1.0
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels on 1d data (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Align HTF indicators to 1d timeframe (completed 1w bar lag)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and EMA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(high_20[i]) or
            np.isnan(low_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 1w trend with volume confirmation
            # Long: price breaks above upper Donchian in uptrend (close > EMA50_1w)
            # Short: price breaks below lower Donchian in downtrend (close < EMA50_1w)
            long_signal = (close[i] > high_20[i]) and (close[i] > ema50_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < low_20[i]) and (close[i] < ema50_aligned[i]) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below lower Donchian (mean reversion)
            exit_signal = close[i] < low_20[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above upper Donchian (mean reversion)
            exit_signal = close[i] > high_20[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrendFilter_VolumeConfirm"
timeframe = "1d"
leverage = 1.0