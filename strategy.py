#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike
- Donchian(20) from prior 1d provides clear breakout levels
- 1w EMA50 > 1w EMA200 ensures alignment with strong weekly trend
- Volume > 1.5x 20-period average confirms breakout momentum
- Designed for 1d timeframe targeting 7-25 trades/year (30-100 over 4 years) to minimize fee drag
- Works in bull markets via breakouts with strong trend, in bear markets via mean reversion at extremes
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
    
    # Get 1d data for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Prior day's high/low for Donchian(20)
    highest_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (completed 1d bar only)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 and EMA200
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMAs to 1d timeframe (completed 1w bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(39, 20)  # Donchian needs 20+19, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout signals with weekly trend filter and volume spike
        # Long: price breaks above Donchian high + weekly uptrend (EMA50>EMA200) + volume spike
        # Short: price breaks below Donchian low + weekly downtrend (EMA50<EMA200) + volume spike
        long_signal = (close[i] > highest_20_aligned[i] and 
                      ema_50_aligned[i] > ema_200_aligned[i] and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (close[i] < lowest_20_aligned[i] and 
                       ema_50_aligned[i] < ema_200_aligned[i] and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend weakening or opposite Donchian level break
            exit_signal = False
            
            if position == 1:
                # Exit long: weekly trend weakening or price breaks below Donchian low
                if (ema_50_aligned[i] <= ema_200_aligned[i] or 
                    close[i] < lowest_20_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: weekly trend weakening or price breaks above Donchian high
                if (ema_50_aligned[i] >= ema_200_aligned[i] or 
                    close[i] > highest_20_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMATrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0