#!/usr/bin/env python3
"""
Hypothesis: 1-hour Stochastic Oscillator with 4-hour trend filter and volume confirmation.
Long when %K crosses above %D in oversold territory (<20), 4h EMA21 trending up, and volume > 1.5x average.
Short when %K crosses below %D in overbought territory (>80), 4h EMA21 trending down, and volume > 1.5x average.
Exit when %K crosses %D in opposite direction or 4h trend reverses.
Uses 4h for trend direction, 1h for precise entry timing. Target 15-35 trades/year.
Works in both bull and bear markets by requiring trend alignment and mean reversion in extreme zones.
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
    
    # Load 4-hour data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour EMA21 for trend direction
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_prev = np.roll(ema21_4h, 1)
    ema21_4h_prev[0] = ema21_4h[0]
    ema21_4h_rising = ema21_4h > ema21_4h_prev
    ema21_4h_falling = ema21_4h < ema21_4h_prev
    
    # Calculate 1-hour Stochastic Oscillator (14,3,3)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    diff = highest_high - lowest_low
    k_percent = np.where(diff != 0, 100 * (close - lowest_low) / diff, 0)
    
    # Smooth %K to get %D (3-period SMA of %K)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1-hour timeframe
    ema21_4h_rising_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h_rising)
    ema21_4h_falling_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema21_4h_rising_aligned[i]) or np.isnan(ema21_4h_falling_aligned[i]) or 
            np.isnan(k_percent[i]) or np.isnan(d_percent[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        k_val = k_percent[i]
        d_val = d_percent[i]
        k_prev = k_percent[i-1]
        d_prev = d_percent[i-1]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        ema21_rising = ema21_4h_rising_aligned[i]
        ema21_falling = ema21_4h_falling_aligned[i]
        
        if position == 0:
            # Long: %K crosses above %D in oversold (<20), 4h EMA21 rising, volume confirmation
            if (k_prev <= d_prev and k_val > d_val and k_val < 20 and 
                ema21_rising and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: %K crosses below %D in overbought (>80), 4h EMA21 falling, volume confirmation
            elif (k_prev >= d_prev and k_val < d_val and k_val > 80 and 
                  ema21_falling and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: %K crosses below %D OR 4h EMA21 starts falling
                if (k_prev >= d_prev and k_val < d_val) or ema21_falling:
                    exit_signal = True
            else:  # position == -1
                # Exit short: %K crosses above %D OR 4h EMA21 starts rising
                if (k_prev <= d_prev and k_val > d_val) or ema21_rising:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Stochastic_4hEMA21_Trend_Volume"
timeframe = "1h"
leverage = 1.0