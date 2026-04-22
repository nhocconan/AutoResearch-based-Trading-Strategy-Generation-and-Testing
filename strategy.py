#!/usr/bin/env python3

"""
Hypothesis: 6-hour 55-period Exponential Moving Average (EMA) crossover with 1-week Ichimoku Cloud filter and volume confirmation.
Long when: 6h EMA(55) crosses above EMA(89) AND price > 1-week Ichimoku Cloud (Senkou Span A/B) AND volume > 1.3x 20-period average.
Short when: 6h EMA(55) crosses below EMA(89) AND price < 1-week Ichimoku Cloud AND volume > 1.3x 20-period average.
Exit when opposite EMA crossover occurs.
Ichimoku Cloud from weekly timeframe provides strong trend filter that works in both bull and bear markets by avoiding counter-trend trades.
Targets 12-37 trades/year (50-150 total over 4 years) with disciplined entry/exit to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for Ichimoku Cloud - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku Components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # The cloud is between Senkou Span A and B
    ichimoku_cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    ichimoku_cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align Ichimoku levels to 6h timeframe
    ichimoku_top_aligned = align_htf_to_ltf(prices, df_1w, ichimoku_cloud_top.values)
    ichimoku_bottom_aligned = align_htf_to_ltf(prices, df_1w, ichimoku_cloud_bottom.values)
    
    # 6h EMA indicators
    ema_fast = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema_slow = pd.Series(close).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(ichimoku_top_aligned[i]) or np.isnan(ichimoku_bottom_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        # EMA crossover signals
        ema_cross_up = (ema_fast[i] > ema_slow[i]) and (ema_fast[i-1] <= ema_slow[i-1])
        ema_cross_down = (ema_fast[i] < ema_slow[i]) and (ema_fast[i-1] >= ema_slow[i-1])
        
        if position == 0 and vol_spike:
            # Long: EMA bullish crossover AND price above Ichimoku cloud
            if ema_cross_up and close[i] > ichimoku_top_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: EMA bearish crossover AND price below Ichimoku cloud
            elif ema_cross_down and close[i] < ichimoku_bottom_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit on opposite EMA crossover
            exit_signal = False
            
            if position == 1 and ema_cross_down:
                exit_signal = True
            elif position == -1 and ema_cross_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_EMA55_89_IchimokuCloud_Volume"
timeframe = "6h"
leverage = 1.0