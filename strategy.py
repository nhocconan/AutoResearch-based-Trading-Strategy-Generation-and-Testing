#!/usr/bin/env python3
"""
4h_TripleBottom_Top_Reversal
Hypothesis: On 4h timeframe, identify triple bottom (support) and triple top (resistance) patterns using 3 consecutive similar lows/highs. Enter long at break above triple bottom resistance with volume confirmation, short at break below triple top support. Uses 1d trend filter (price vs 200 EMA) to align with higher timeframe bias. Designed to work in both bull (buy dips) and bear (sell rallies) markets by waiting for clear reversal patterns with volume confirmation, reducing false signals in chop. Target: 20-35 trades/year.
"""

name = "4h_TripleBottom_Top_Reversal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Get 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    # Detect triple bottom: 3 similar lows within 2% range
    triple_bottom = np.zeros(n, dtype=bool)
    triple_top = np.zeros(n, dtype=bool)
    
    lookback = 50  # Look for pattern within last 50 bars (~8 days)
    
    for i in range(50, n):
        # Skip if insufficient data for pattern detection
        if i < lookback:
            continue
            
        # Check for triple bottom in last 50 bars
        window_low = low[i-lookback:i]
        # Find local minima
        mins = []
        for j in range(1, len(window_low)-1):
            if window_low[j] <= window_low[j-1] and window_low[j] <= window_low[j+1]:
                mins.append((i-lookback+j, window_low[j]))
        
        # Check if we have 3 similar lows
        if len(mins) >= 3:
            # Take last 3 minima
            last_three = mins[-3:]
            low_vals = [x[1] for x in last_three]
            low_range = max(low_vals) - min(low_vals)
            avg_low = sum(low_vals) / 3
            if avg_low > 0 and (low_range / avg_low) < 0.02:  # Within 2%
                # Resistance level is the high between the lows
                # Find highest point between first and third low
                idx1, idx3 = last_three[0][0], last_three[2][0]
                if idx3 > idx1:
                    resistance = np.max(high[idx1:idx3+1])
                    triple_bottom[i] = True
                    # Store resistance level for breakout check
                    if not hasattr(generate_signals, 'resistance_level'):
                        generate_signals.resistance_level = np.full(n, np.nan)
                    generate_signals.resistance_level[i] = resistance
        
        # Check for triple top in last 50 bars
        window_high = high[i-lookback:i]
        # Find local maxima
        maxs = []
        for j in range(1, len(window_high)-1):
            if window_high[j] >= window_high[j-1] and window_high[j] >= window_high[j+1]:
                maxs.append((i-lookback+j, window_high[j]))
        
        # Check if we have 3 similar highs
        if len(maxs) >= 3:
            # Take last 3 maxima
            last_three = maxs[-3:]
            high_vals = [x[1] for x in last_three]
            high_range = max(high_vals) - min(high_vals)
            avg_high = sum(high_vals) / 3
            if avg_high > 0 and (high_range / avg_high) < 0.02:  # Within 2%
                # Support level is the low between the highs
                idx1, idx3 = last_three[0][0], last_three[2][0]
                if idx3 > idx1:
                    support = np.min(low[idx1:idx3+1])
                    triple_top[i] = True
                    if not hasattr(generate_signals, 'support_level'):
                        generate_signals.support_level = np.full(n, np.nan)
                    generate_signals.support_level[i] = support
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical values are NaN
        if np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1d EMA200
        uptrend_1d = close[i] > ema200_1d_aligned[i]
        downtrend_1d = close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: break above triple bottom resistance with volume filter in uptrend
            if triple_bottom[i] and hasattr(generate_signals, 'resistance_level'):
                resistance = getattr(generate_signals, 'resistance_level')[i]
                if not np.isnan(resistance) and high[i] > resistance and uptrend_1d and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: break below triple top support with volume filter in downtrend
            elif triple_top[i] and hasattr(generate_signals, 'support_level'):
                support = getattr(generate_signals, 'support_level')[i]
                if not np.isnan(support) and low[i] < support and downtrend_1d and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price drops below triple bottom support or trend fails
            if triple_top[i] and hasattr(generate_signals, 'support_level'):
                support = getattr(generate_signals, 'support_level')[i]
                if not np.isnan(support) and low[i] < support:
                    signals[i] = 0.0
                    position = 0
            elif not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above triple top resistance or trend fails
            if triple_bottom[i] and hasattr(generate_signals, 'resistance_level'):
                resistance = getattr(generate_signals, 'resistance_level')[i]
                if not np.isnan(resistance) and high[i] > resistance:
                    signals[i] = 0.0
                    position = 0
            elif not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals