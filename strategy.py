#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation
- Donchian breakout provides clear structure-based entries in both bull and bear markets
- 1d EMA(50) ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume confirmation (>1.5x 20-period average) filters false breakouts
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years) to minimize fee drag
- Uses discrete position sizing (0.0, ±0.25) to reduce churn
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 4h
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, 20)  # EMA1d, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: price breaks above upper Donchian + uptrend + volume spike
        # Short: price breaks below lower Donchian + downtrend + volume spike
        long_signal = (close[i] > high_ma[i] and 
                      close[i] > ema_50_1d_aligned[i] and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (close[i] < low_ma[i] and 
                       close[i] < ema_50_1d_aligned[i] and
                       volume[i] > 1.5 * vol_ma[i])
        
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
                if (close[i] < ema_50_1d_aligned[i] or 
                    close[i] < low_ma[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above upper Donchian
                if (close[i] > ema_50_1d_aligned[i] or 
                    close[i] > high_ma[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0