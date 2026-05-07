#!/usr/bin/env python3
"""
12H_Triple_Crossover_Volume_Squeeze_Exit_v1
Hypothesis: On 12h timeframe, use a triple EMA crossover system (8/21/55) for trend direction,
with volume confirmation and Bollinger Band squeeze exit. The triple EMA provides robust
trend filtering, volume confirms institutional participation, and BB squeeze identifies
low volatility breakouts. Works in both bull and bear markets by capturing strong
trends while avoiding choppy conditions.
"""
name = "12H_Triple_Crossover_Volume_Squeeze_Exit_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (avoid false signals in weak trends)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate 1d EMA55 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema55_1d = close_1d.ewm(span=55, adjust=False, min_periods=55).mean().values
    ema55_1d_aligned = align_htf_to_ltf(prices, df_1d, ema55_1d)
    
    # Get 12h data for EMA8/EMA21 crossover
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h EMA8 and EMA21
    close_12h = pd.Series(df_12h['close'])
    ema8_12h = close_12h.ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_12h = close_12h.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema8_12h_aligned = align_htf_to_ltf(prices, df_12h, ema8_12h)
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Bollinger Bands on 12h for squeeze detection (exit signal)
    sma20_12h = close_12h.rolling(window=20, min_periods=20).mean().values
    std20_12h = close_12h.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20_12h + (2 * std20_12h)
    lower_bb = sma20_12h - (2 * std20_12h)
    bb_width = (upper_bb - lower_bb) / sma20_12h
    bb_width_aligned = align_htf_to_ltf(prices, df_12h, bb_width)
    
    # Volume filter: current volume > 1.3x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(55, 20)  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema8_12h_aligned[i]) or np.isnan(ema21_12h_aligned[i]) or 
            np.isnan(ema55_1d_aligned[i]) or np.isnan(bb_width_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (12 days on 12h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Bullish trend: EMA8 > EMA21 > EMA55(1d)
            bullish_trend = (ema8_12h_aligned[i] > ema21_12h_aligned[i] and 
                           ema21_12h_aligned[i] > ema55_1d_aligned[i])
            
            # Bearish trend: EMA8 < EMA21 < EMA55(1d)
            bearish_trend = (ema8_12h_aligned[i] < ema21_12h_aligned[i] and 
                           ema21_12h_aligned[i] < ema55_1d_aligned[i])
            
            # Long: bullish crossover + volume confirmation
            if (bullish_trend and 
                ema8_12h_aligned[i-1] <= ema21_12h_aligned[i-1] and  # crossover just happened
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: bearish crossover + volume confirmation
            elif (bearish_trend and 
                  ema8_12h_aligned[i-1] >= ema21_12h_aligned[i-1] and  # crossover just happened
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            # Exit 1: Bollinger Band squeeze (low volatility breakout fading)
            if bb_width_aligned[i] < 0.02:  # Very tight Bollinger Bands
                exit_signal = True
            
            # Exit 2: Trend reversal (EMA8 crosses back through EMA21)
            elif position == 1 and ema8_12h_aligned[i] < ema21_12h_aligned[i]:
                exit_signal = True
            elif position == -1 and ema8_12h_aligned[i] > ema21_12h_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals