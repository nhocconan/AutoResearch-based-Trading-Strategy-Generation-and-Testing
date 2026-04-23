#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Long when price breaks above Donchian(20) high AND weekly pivot > weekly close (bullish bias) AND volume > 1.8x 20-period average.
Short when price breaks below Donchian(20) low AND weekly pivot < weekly close (bearish bias) AND volume > 1.8x 20-period average.
Exit when price touches the opposite Donchian level or reverses Donchian direction.
Weekly pivot provides structural bias from higher timeframe to avoid counter-trend trades in ranging markets.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Calculate weekly pivot for bias filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_bullish = weekly_pivot > weekly_close  # Bullish bias when pivot above close
    
    # Align weekly bias to 6h timeframe (use previous week's bias for current week)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float), additional_delay_bars=1)
    
    # Calculate Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Donchian(20) and volume MA need 20 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_bullish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        bullish_bias = weekly_bullish_aligned[i] > 0.5  # Convert back to boolean
        bearish_bias = weekly_bullish_aligned[i] <= 0.5
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above Donchian high AND weekly bullish bias AND volume spike
            if price > high_20[i] and bullish_bias and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND weekly bearish bias AND volume spike
            elif price < low_20[i] and bearish_bias and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Donchian low OR weekly bias turns bearish
                if price < low_20[i] or bearish_bias:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Donchian high OR weekly bias turns bullish
                if price > high_20[i] or bullish_bias:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_WeeklyPivotBias_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0