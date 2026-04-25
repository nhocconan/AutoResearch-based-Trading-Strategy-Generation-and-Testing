#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Trade 4h Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation.
- Trend filter: price > 1d EMA50 = bullish, price < 1d EMA50 = bearish.
- In bullish 1d trend: buy breakouts above upper Donchian(20), sell breakdowns below lower Donchian(20).
- In bearish 1d trend: sell breakdowns below lower Donchian(20), buy breakouts above upper Donchian(20) (continuation logic).
- Volume confirmation: require volume > 1.5x 20-period average to avoid false breakouts.
- Position size: 0.25. Target: 75-200 total trades over 4 years = 19-50/year.
- Works in both bull and bear: 1d trend filter captures major moves, volume filters noise, Donchian provides objective breakout levels.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels on primary timeframe (4h)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume spike confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20) and volume MA (20)
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend using EMA50
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Breakout logic: trade in direction of 1d trend
            long_setup = (close[i] > upper[i]) and htf_1d_bullish and volume_spike[i]
            short_setup = (close[i] < lower[i]) and htf_1d_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on trend reversal or mean reversion to midpoint
            midpoint = (upper[i] + lower[i]) / 2.0
            exit_signal = (not htf_1d_bullish) or (close[i] < midpoint)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal or mean reversion to midpoint
            midpoint = (upper[i] + lower[i]) / 2.0
            exit_signal = htf_1d_bullish or (close[i] > midpoint)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0