#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) Breakout with 12h EMA Trend Filter and Volume Confirmation
- Donchian(20) provides clear structure for breakouts in both bull and bear markets
- 12h EMA(50) trend filter ensures alignment with intermediate-term trend direction
- Volume > 2.0x 20-period average confirms breakout momentum with conviction
- Designed for 4h timeframe targeting 25-40 trades/year (100-160 over 4 years)
- Works in bull markets via breakouts with trend, in bear markets via mean reversion at channel extremes
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
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe (completed bars only)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    close_12h = df_12h['close'].values
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA needs 50 bars, Donchian 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend direction from 12h EMA
        uptrend = close_12h_aligned[i] > ema_50_aligned[i]
        downtrend = close_12h_aligned[i] < ema_50_aligned[i]
        
        # Donchian breakout signals with trend filter and volume spike
        # Long: price breaks above upper Donchian + uptrend + volume spike
        # Short: price breaks below lower Donchian + downtrend + volume spike
        long_signal = (close[i] > high_ma[i] and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < low_ma[i] and 
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
            # Exit conditions: trend reversal or opposite Donchian level break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend turns down or price breaks below lower Donchian
                if (not uptrend or 
                    close[i] < low_ma[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend turns up or price breaks above upper Donchian
                if (not downtrend or 
                    close[i] > high_ma[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0