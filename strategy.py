#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
Weekly pivot direction provides structural bias (bull/bear) from higher timeframe.
Long when price breaks above 6h Donchian upper band AND weekly pivot bias is bullish AND volume > 2.0x 20-period MA.
Short when price breaks below 6h Donchian lower band AND weekly pivot bias is bearish AND volume > 2.0x 20-period MA.
Exit when price returns to 6h Donchian middle band (mean reversion) or opposite breakout occurs.
Designed for ~12-25 trades/year with structural edge from weekly pivot filtering false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w HTF data for weekly pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot: based on previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot point and bias
    # Pivot = (High + Low + Close) / 3
    # Bias bullish if close > pivot, bearish if close < pivot
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    weekly_bias_bullish = close_1w > pivot_1w
    weekly_bias_bearish = close_1w < pivot_1w
    
    # Align weekly bias to 6h timeframe (completed weekly bars only)
    weekly_bias_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bullish.astype(float))
    weekly_bias_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bearish.astype(float))
    
    # Calculate 6h Donchian channels (20-period)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    # Middle band = (upper + lower) / 2
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)  # need Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_bias_bullish_aligned[i]) or 
            np.isnan(weekly_bias_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly pivot bias filters
        bias_bullish = weekly_bias_bullish_aligned[i] > 0.5
        bias_bearish = weekly_bias_bearish_aligned[i] > 0.5
        
        # Volume filter: 6h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i]  # Break above upper band
        breakout_down = close[i] < donchian_lower[i]  # Break below lower band
        return_to_middle = abs(close[i] - donchian_middle[i]) < 0.1 * (donchian_upper[i] - donchian_lower[i])  # Near middle band
        opposite_breakout = (position == 1 and breakout_down) or \
                            (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above upper band AND bullish weekly bias AND volume confirmation
            if breakout_up and bias_bullish and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND bearish weekly bias AND volume confirmation
            elif breakout_down and bias_bearish and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to middle band or opposite breakout
            exit_signal = False
            if position == 1:
                exit_signal = return_to_middle or opposite_breakout
            elif position == -1:
                exit_signal = return_to_middle or opposite_breakout
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_WeeklyPivotBias_VolumeSpike"
timeframe = "6h"
leverage = 1.0