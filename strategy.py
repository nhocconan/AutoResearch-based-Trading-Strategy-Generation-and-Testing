#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend direction.
- EMA > price indicates uptrend bias for longs; EMA < price indicates downtrend bias for shorts.
- Entry: Long when price breaks above Donchian(20) upper AND price > 1d EMA34.
         Short when price breaks below Donchian(20) lower AND price < 1d EMA34.
- Exit: Opposite Donchian breakout.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA (34-period) on 1d
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, lookback, 20)  # Need enough 1d bars for EMA and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                # Bullish breakout: price closes above upper Donchian AND price > 1d EMA (uptrend bias)
                if curr_close > highest_high[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below lower Donchian AND price < 1d EMA (downtrend bias)
                elif curr_close < lowest_low[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below Donchian lower
            if curr_close < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian upper
            if curr_close > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0