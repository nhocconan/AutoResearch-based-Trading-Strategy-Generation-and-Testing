#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
- Long: Close breaks above Donchian(20) high + price > 12h EMA50 (uptrend) + volume > 2.0x 20-period average
- Short: Close breaks below Donchian(20) low + price < 12h EMA50 (downtrend) + volume > 2.0x 20-period average
- Exit: Opposite Donchian breakout (close < Donchian(20) low for long, close > Donchian(20) high for short)
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag
- Donchian channels provide clear structure; volume confirms institutional participation; 12h EMA50 filters counter-trend noise
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 12h EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: Close breaks above Donchian high + uptrend + volume spike
        # Short: Close breaks below Donchian low + downtrend + volume spike
        long_signal = (close[i] > donchian_high[i] and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < donchian_low[i] and 
                       downtrend and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Opposite Donchian breakout
            exit_signal = False
            
            if position == 1:
                # Exit long: Close breaks below Donchian low
                if close[i] < donchian_low[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Close breaks above Donchian high
                if close[i] > donchian_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0