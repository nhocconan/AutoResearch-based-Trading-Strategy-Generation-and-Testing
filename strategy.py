#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d Williams %R regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for Williams %R overbought/oversold conditions.
- Williams %R < -80 indicates oversold (bullish bias), > -20 indicates overbought (bearish bias).
- Entry: Long when price breaks above Donchian(20) upper AND Williams %R < -80 (breakout from oversold).
         Short when price breaks below Donchian(20) lower AND Williams %R > -20 (breakout from overbought).
         In neutral (-80 <= %R <= -20): Long on pullback to Donchian mid with bullish candle.
                                      Short on pullback to Donchian mid with bearish candle.
- Exit: Opposite Donchian breakout or Williams %R reaching opposite extreme.
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
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 1d
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    willr = np.where(rr != 0, -100 * (highest_high - close_1d) / rr, -50.0)
    
    # Align 1d Williams %R to 4h
    willr_aligned = align_htf_to_ltf(prices, df_1d, willr)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high_4h = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low_4h = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high_4h + lowest_low_4h) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, lookback, 20)  # Need enough 1d bars for Williams %R and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(willr_aligned[i]) or np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        willr_val = willr_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if willr_val < -80:  # Oversold: bullish bias
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high_4h[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bullish pullback: price touches Donchian mid with bullish candle
                    elif curr_low <= donchian_mid[i] and curr_close > prev_close:
                        signals[i] = 0.25
                        position = 1
                elif willr_val > -20:  # Overbought: bearish bias
                    # Bearish breakout: price closes below lower Donchian
                    if curr_close < lowest_low_4h[i]:
                        signals[i] = -0.25
                        position = -1
                    # Bearish pullback: price touches Donchian mid with bearish candle
                    elif curr_high >= donchian_mid[i] and curr_close < prev_close:
                        signals[i] = -0.25
                        position = -1
                else:  # Neutral: wait for clear signal
                    pass
        elif position == 1:
            # Long exit: price closes below Donchian lower OR Williams %R reaches overbought
            if curr_close < lowest_low_4h[i] or willr_val > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian upper OR Williams %R reaches oversold
            if curr_close > highest_high_4h[i] or willr_val < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dWilliamsR_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0