#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) Breakout with 1d EMA50 Trend and Volume Confirmation
- Donchian(20) on 12h captures intermediate-term price channels and breakouts
- 1d EMA(50) ensures alignment with daily trend for multi-timeframe confirmation
- Volume > 1.8x 20-period average confirms strong breakout momentum
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in bull markets via breakouts with trend, in bear markets via mean reversion at extremes
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
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
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
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels on 12h timeframe using lookback of 20 periods
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
            
            # Donchian breakout signals with trend filter and volume confirmation
            # Long: price breaks above upper Donchian + uptrend + volume confirmation
            # Short: price breaks below lower Donchian + downtrend + volume confirmation
            long_signal = (close[i] > highest_high and 
                          close[i] > ema_50_1d_aligned[i] and
                          volume[i] > 1.8 * vol_ma[i])
            
            short_signal = (close[i] < lowest_low and 
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
                # Exit conditions: trend reversal or opposite Donchian break
                exit_signal = False
                
                if position == 1:
                    # Exit long: trend reversal or price breaks below lower Donchian
                    if (close[i] < ema_50_1d_aligned[i] or 
                        close[i] < lowest_low):
                        exit_signal = True
                elif position == -1:
                    # Exit short: trend reversal or price breaks above upper Donchian
                    if (close[i] > ema_50_1d_aligned[i] or 
                        close[i] > highest_high):
                        exit_signal = True
                
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
        else:
            # Not enough data for Donchian calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0