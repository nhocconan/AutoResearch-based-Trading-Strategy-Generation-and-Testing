#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Trend Filter (EMA200)
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when Bull Power > 0, Bear Power < 0, and price > 1d EMA200 (bullish trend)
# - Short when Bear Power > 0, Bull Power < 0, and price < 1d EMA200 (bearish trend)
# - Exit when Elder Ray signals weaken or price crosses EMA200
# - Designed to capture momentum in trending markets while avoiding counter-trend trades
# - Works in bull markets (long bias) and bear markets (short bias) via 1d EMA200 filter
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_ElderRay_EMA200_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA200 calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA200 on 1d close
    close_1d = df_1d['close']
    ema200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Look for long entry: Bull Power > 0, Bear Power < 0, price > 1d EMA200
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Look for short entry: Bear Power > 0, Bull Power < 0, price < 1d EMA200
            elif (bear_power[i] > 0 and 
                  bull_power[i] < 0 and 
                  close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when Elder Ray weakens or price crosses below EMA200
            if (bull_power[i] <= 0 or 
                bear_power[i] >= 0 or 
                close[i] < ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when Elder Ray weakens or price crosses above EMA200
            if (bear_power[i] <= 0 or 
                bull_power[i] >= 0 or 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals