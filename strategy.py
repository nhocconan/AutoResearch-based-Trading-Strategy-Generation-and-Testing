#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Donchian channel breakouts capture momentum in trending markets
- 1w EMA(50) ensures alignment with weekly trend to avoid counter-trend trades
- Volume confirmation (>2.0x 20-period average) filters false breakouts
- Designed for 1d timeframe targeting 7-25 trades/year (30-100 over 4 years) to minimize fee drag
- Works in both bull and bear markets by trading breakouts in direction of weekly trend
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) on 1d (using rolling window)
    # Upper channel: highest high over past 20 days
    # Lower channel: lowest low over past 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, 20)  # EMA1w, Donchian20, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: price breaks above upper Donchian + uptrend (price > weekly EMA) + volume spike
        # Short: price breaks below lower Donchian + downtrend (price < weekly EMA) + volume spike
        long_signal = (close[i] > donchian_upper[i] and 
                      close[i] > ema_50_1w_aligned[i] and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < donchian_lower[i] and 
                       close[i] < ema_50_1w_aligned[i] and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite Donchian break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or price breaks below lower Donchian
                if (close[i] < ema_50_1w_aligned[i] or 
                    close[i] < donchian_lower[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above upper Donchian
                if (close[i] > ema_50_1w_aligned[i] or 
                    close[i] > donchian_upper[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0