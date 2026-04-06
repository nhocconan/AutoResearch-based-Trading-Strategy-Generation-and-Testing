#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h trend filter
# Long when Bull Power > 0 AND Bear Power < 0 AND EMA(50) > EMA(100) (12h)
# Short when Bear Power < 0 AND Bull Power < 0 AND EMA(50) < EMA(100) (12h)
# Exit when Bull Power and Bear Power have same sign (both positive or both negative)
# Uses Elder Ray to measure bull/bear power relative to EMA(13), 12h EMA crossover for trend filter
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 6h performance

name = "6h_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray Index components (13-period EMA as base)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12h EMA(50) and EMA(100) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMAs on 12h close
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema100_12h = close_12h_series.ewm(span=100, min_periods=100, adjust=False).mean().values
    
    # Align 12h EMAs to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema100_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(120, n):
        # Skip if required data not available
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(ema100_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exit: Bull Power and Bear Power have same sign (both positive or both negative)
        if position == 1:  # long position
            if bull_power[i] > 0 and bear_power[i] > 0:  # both positive
                signals[i] = 0.0
                position = 0
            elif bull_power[i] < 0 and bear_power[i] < 0:  # both negative
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bull_power[i] > 0 and bear_power[i] > 0:  # both positive
                signals[i] = 0.0
                position = 0
            elif bull_power[i] < 0 and bear_power[i] < 0:  # both negative
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Elder Ray signals and 12h trend filter
            # Long: Bull Power > 0 AND Bear Power < 0 AND EMA50 > EMA100 (12h uptrend)
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                ema50_12h_aligned[i] > ema100_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 AND EMA50 < EMA100 (12h downtrend)
            elif (bull_power[i] < 0 and bear_power[i] < 0 and 
                  ema50_12h_aligned[i] < ema100_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals