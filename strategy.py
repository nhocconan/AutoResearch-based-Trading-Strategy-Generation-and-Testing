#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d Williams %R extreme filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for Williams %R overbought/oversold conditions.
- Williams %R < -80 indicates oversold (bullish), > -20 indicates overbought (bearish).
- Entry: Long when price breaks above Donchian(20) upper AND Williams %R < -80 (bullish breakout from oversold).
         Short when price breaks below Donchian(20) lower AND Williams %R > -20 (bearish breakout from overbought).
         In neutral Williams %R (-80 to -20): No new entries, only hold existing positions.
- Exit: Opposite Donchian breakout or Williams %R returns to neutral zone.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 1d
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # Align 1d Williams %R to 12h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Donchian channels (20-period) on 12h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, lookback, 20)  # Need enough 1d bars for Williams %R and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_val = williams_r_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if williams_val < -80:  # Oversold: look for bullish breakout
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high[i]:
                        signals[i] = 0.25
                        position = 1
                elif williams_val > -20:  # Overbought: look for bearish breakout
                    # Bearish breakout: price closes below lower Donchian
                    if curr_close < lowest_low[i]:
                        signals[i] = -0.25
                        position = -1
                # Neutral zone (-80 to -20): no new entries
        elif position == 1:
            # Long exit: price closes below Donchian mid OR Williams %R returns to neutral (> -80)
            if curr_close < donchian_mid[i] or williams_val > -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR Williams %R returns to neutral (< -20)
            if curr_close > donchian_mid[i] or williams_val < -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dWilliamsR_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0