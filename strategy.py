#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with weekly trend filter and volume confirmation.
# Uses weekly EMA13 for trend (bull/bear regime) and Elder Ray (bull/bear power) for entry timing.
# In bull regime (price > weekly EMA13): enter long when bear power crosses above zero.
# In bear regime (price < weekly EMA13): enter short when bull power crosses below zero.
# Weekly trend filter reduces whipsaw, Elder Ray provides timely entries.
# Target: 20-35 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA13 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema13_1w = close_1w_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 13-period EMA for Elder Ray (on 6h data)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High minus EMA13
    bear_power = low - ema13   # Low minus EMA13
    
    # Align weekly EMA13 to 6h
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need weekly EMA13 + EMA13 + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13_1w_aligned[i]) or 
            np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from weekly EMA13
        bull_regime = close[i] > ema13_1w_aligned[i]
        bear_regime = close[i] < ema13_1w_aligned[i]
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: Bull regime AND bear power crosses above zero (selling pressure weakening)
            if bull_regime and (bear_power[i] > 0) and (bear_power[i-1] <= 0) and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear regime AND bull power crosses below zero (buying pressure weakening)
            elif bear_regime and (bull_power[i] < 0) and (bull_power[i-1] >= 0) and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear power crosses below zero OR regime turns bearish
            if (bear_power[i] < 0 and bear_power[i-1] >= 0) or bear_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull power crosses above zero OR regime turns bullish
            if (bull_power[i] > 0 and bull_power[i-1] <= 0) or bull_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_WeeklyEMA13_Volume"
timeframe = "6h"
leverage = 1.0