#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_camarilla_ema50_v1
# Uses 12h Camarilla levels (H3/L3) with 4h EMA50 filter for trend direction.
# Long when price > EMA50 and breaks above H3; Short when price < EMA50 and breaks below L3.
# Volume confirmation: volume > 1.5x 20-period average.
# This reduces false signals by aligning with higher timeframe trend.
# Target: 20-40 trades/year per symbol.

name = "4h_12h_camarilla_ema50_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    high_prev = df_12h['high'].shift(1).values
    low_prev = df_12h['low'].shift(1).values
    close_prev = df_12h['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align to 4h timeframe
    h3_level = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # EMA50 on 4h close
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if levels not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or np.isnan(ema50[i]):
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price > EMA50 and breaks above H3
        if close[i] > ema50[i] and close[i] > h3_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price < EMA50 and breaks below L3
        elif close[i] < ema50[i] and close[i] < l3_level[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: reverse crossover of EMA50
        elif close[i] < ema50[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > ema50[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals