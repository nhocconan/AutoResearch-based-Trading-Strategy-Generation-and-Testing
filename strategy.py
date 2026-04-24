#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d Williams %R regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for Williams %R overbought/oversold conditions.
- Williams %R > -20 indicates overbought (favor shorts on breakdowns), Williams %R < -80 indicates oversold (favor longs on breakouts).
- Entry: Long when price breaks above Donchian(20) upper AND Williams %R < -80 (breakout from oversold).
         Short when price breaks below Donchian(20) lower AND Williams %R > -20 (breakdown from overbought).
         In neutral (-80 <= Williams %R <= -20): require stricter volume confirmation (1.5x average).
- Exit: Opposite Donchian breakout or Williams %R crosses midpoint (-50) indicating regime shift.
- Volume confirmation: current volume > 1.3 * 20-period volume MA (to avoid false breakouts).
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
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 1d
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - pd.Series(df_1d['close'])) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # Align 1d Williams %R to 4h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high_4h = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low_4h = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high_4h + lowest_low_4h) / 2.0
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, lookback, 20)  # Need enough 1d bars for Williams %R and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr_val = williams_r_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if wr_val < -80:  # Oversold: favor longs on breakouts
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high_4h[i]:
                        signals[i] = 0.25
                        position = 1
                elif wr_val > -20:  # Overbought: favor shorts on breakdowns
                    # Bearish breakout: price closes below lower Donchian
                    if curr_close < lowest_low_4h[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Neutral regime: require stricter volume confirmation
                    if volume > (1.5 * volume_ma[i]):  # Stricter volume filter
                        # Bullish breakout: price closes above upper Donchian
                        if curr_close > highest_high_4h[i]:
                            signals[i] = 0.25
                            position = 1
                        # Bearish breakout: price closes below lower Donchian
                        elif curr_close < lowest_low_4h[i]:
                            signals[i] = -0.25
                            position = -1
        elif position == 1:
            # Long exit: price closes below Donchian mid OR Williams %R crosses above -50 (shift from oversold)
            if curr_close < donchian_mid[i] or wr_val > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR Williams %R crosses below -50 (shift from overbought)
            if curr_close > donchian_mid[i] or wr_val < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dWilliamsRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0