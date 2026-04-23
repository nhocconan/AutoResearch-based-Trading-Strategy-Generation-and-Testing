#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
- Donchian(20) from 1d HTF provides robust structure-based breakout levels
- 1d EMA(50) ensures alignment with higher timeframe trend for multi-timeframe confirmation
- Volume > 1.8x 20-period average confirms strong breakout momentum and reduces false signals
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in bull markets via breakouts with trend, in bear markets via fade of overextended moves at strong levels
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
    
    # Get 1d data for Donchian channels and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Donchian calculation (20-period)
    # Need at least 20 days of data
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on daily data
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (completed 1d bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Get 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA1d, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: price breaks above Donchian high + uptrend + volume confirmation
        # Short: price breaks below Donchian low + downtrend + volume confirmation
        long_signal = (close[i] > donchian_high_aligned[i] and 
                      close[i] > ema_50_1d_aligned[i] and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < donchian_low_aligned[i] and 
                       close[i] < ema_50_1d_aligned[i] and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite Donchian level break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or price breaks below Donchian low
                if (close[i] < ema_50_1d_aligned[i] or 
                    close[i] < donchian_low_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above Donchian high
                if (close[i] > ema_50_1d_aligned[i] or 
                    close[i] > donchian_high_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0