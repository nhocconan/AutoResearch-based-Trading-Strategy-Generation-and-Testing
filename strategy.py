#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h trend filter
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and Bear Power < 0 AND price > 12h EMA50 (bullish regime)
# Short when Bull Power < 0 and Bear Power > 0 AND price < 12h EMA50 (bearish regime)
# Exit when Bull Power and Bear Power have same sign (both positive or both negative)
# Uses 12h EMA for regime filter to avoid counter-trend trades
# Target: 75-150 total trades over 4 years (19-38/year) for optimal 6h performance

name = "6h_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray components: EMA13 of close
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_s = pd.Series(close_12h)
    ema50_12h = close_12h_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exit: Bull Power and Bear Power have same sign (both + or both -)
        if position == 1:  # long position
            if bull_power[i] > 0 and bear_power[i] > 0:  # both positive = overextended
                signals[i] = 0.0
                position = 0
            elif bull_power[i] < 0 and bear_power[i] < 0:  # both negative = weakness
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bull_power[i] > 0 and bear_power[i] > 0:  # both positive = strength
                signals[i] = 0.0
                position = 0
            elif bull_power[i] < 0 and bear_power[i] < 0:  # both negative = overextended short
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter
            # Long: Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA50
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 AND price < 12h EMA50
            elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals