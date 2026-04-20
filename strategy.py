#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1-day EMA200 filter
# Uses bull power (high - EMA13) and bear power (EMA13 - low) with 13-period EMA
# Long when bull power > 0 and bear power increasing (momentum) and price > daily EMA200
# Short when bear power < 0 and bull power decreasing and price < daily EMA200
# Exits when power signals reverse or price crosses EMA13
# Works in both bull/bear markets by focusing on institutional buying/selling pressure
# Target: 15-35 trades/year (60-140 total over 4 years)

name = "6h_ElderRay_EMA200_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for EMA200 filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Buying strength
    bear_power = ema13 - low   # Selling strength
    
    # Calculate EMA of bull/bear power for momentum (13-period)
    bull_power_ema = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_ema = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull power momentum: current > EMA (increasing)
    bull_momentum = bull_power > bull_power_ema
    # Bear power momentum: current > EMA (increasing selling pressure)
    bear_momentum = bear_power > bear_power_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if any value is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        ema13_val = ema13[i]
        ema200_val = ema200_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        bull_mom = bull_momentum[i]
        bear_mom = bear_momentum[i]
        
        if position == 0:
            # Long: bull power positive AND increasing AND price above daily EMA200
            if bull_val > 0 and bull_mom and close_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: bear power positive AND increasing AND price below daily EMA200
            elif bear_val > 0 and bear_mom and close_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bull power turns negative OR price crosses below EMA13
            if bull_val <= 0 or close_val < ema13_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bear power turns negative OR price crosses above EMA13
            if bear_val <= 0 or close_val > ema13_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals