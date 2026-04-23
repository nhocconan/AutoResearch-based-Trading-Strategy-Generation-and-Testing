#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
- Long when price breaks above Donchian(20) high + price above 1d EMA50 + volume > 1.5x 20-period average
- Short when price breaks below Donchian(20) low + price below 1d EMA50 + volume > 1.5x 20-period average
- Uses 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in bull markets via trend continuation breakouts, in bear markets via breakdown shorts
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
    
    # Get 1d data for EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 12h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Donchian(20) on 12h timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian needs 20, EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_roll[i-1]  # Break above previous period's high
        short_breakout = close[i] < low_roll[i-1]  # Break below previous period's low
        
        # Trend filter from 1d EMA50
        uptrend = close_1d_aligned[i] > ema50_aligned[i]
        downtrend = close_1d_aligned[i] < ema50_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            if long_breakout and uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            elif short_breakout and downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low or trend turns down
                if (close[i] < low_roll[i-1] or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian high or trend turns up
                if (close[i] > high_roll[i-1] or 
                    not downtrend):
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