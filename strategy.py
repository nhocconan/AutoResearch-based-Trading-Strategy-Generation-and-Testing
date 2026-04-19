# State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
# Hypothesis: 4h Donchian(20) breakout with 1-day volume confirmation and 1-week EMA50 trend filter.
# Donchian breakouts capture trend continuation; volume confirms institutional participation.
# EMA50 on weekly timeframe acts as a robust trend filter to avoid counter-trend trades.
# Works in bull markets by catching breakouts; works in bear markets by filtering out false breakouts during downtrends.
# Designed for 4h timeframe to balance trade frequency and signal quality.
# Entry: Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND weekly EMA50 rising.
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND weekly EMA50 falling.
# Exit: Opposite Donchian level touch or EMA50 direction change.
# Uses strict conditions to limit trades (~20-40/year) and avoid overtrading.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Volume_EMA50_WeeklyTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    # Weekly EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly EMA50 direction: rising if current > previous, falling if current < previous
    ema50_prev = np.roll(ema50_1w_aligned, 1)
    ema50_prev[0] = ema50_1w_aligned[0]  # handle first value
    ema50_rising = ema50_1w_aligned > ema50_prev
    ema50_falling = ema50_1w_aligned < ema50_prev
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume and weekly uptrend
            if (close[i] > high_20[i] and 
                volume_confirm[i] and 
                ema50_rising[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and weekly downtrend
            elif (close[i] < low_20[i] and 
                  volume_confirm[i] and 
                  ema50_falling[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches Donchian low or weekly EMA turns down
            if (close[i] < low_20[i]) or (not ema50_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches Donchian high or weekly EMA turns up
            if (close[i] > high_20[i]) or (not ema50_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals