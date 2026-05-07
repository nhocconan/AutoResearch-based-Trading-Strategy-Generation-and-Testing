#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Fibonacci retracement levels from 1d swing high/low + volume confirmation.
# Long when price retraces to 61.8% level of 1d uptrend and volume spikes.
# Short when price retraces to 38.2% level of 1d downtrend and volume spikes.
# Uses 1d swing points to identify trend and Fibonacci levels for mean reversion.
# Volume confirms momentum at key levels. Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
name = "6h_Fib_Retracement_1dSwing_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for swing high/low (lookback 20 periods)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d swing high (max high over 20 periods) and swing low (min low over 20 periods)
    swing_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    swing_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align swing levels to 6m timeframe
    swing_high_1d_aligned = align_htf_to_ltf(prices, df_1d, swing_high_1d)
    swing_low_1d_aligned = align_htf_to_ltf(prices, df_1d, swing_low_1d)
    
    # Calculate Fibonacci retracement levels: 38.2% and 61.8%
    diff_1d = swing_high_1d_aligned - swing_low_1d_aligned
    fib_382 = swing_low_1d_aligned + 0.382 * diff_1d  # 38.2% level
    fib_618 = swing_low_1d_aligned + 0.618 * diff_1d  # 61.8% level
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for swing calculations
    
    for i in range(start_idx, n):
        if (np.isnan(swing_high_1d_aligned[i]) or np.isnan(swing_low_1d_aligned[i]) or 
            np.isnan(fib_382[i]) or np.isnan(fib_618[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near 61.8% retracement level (support in uptrend) + volume spike
            near_618 = np.abs(close[i] - fib_618[i]) <= (0.005 * close[i])  # Within 0.5% of level
            # Short: price near 38.2% retracement level (resistance in downtrend) + volume spike
            near_382 = np.abs(close[i] - fib_382[i]) <= (0.005 * close[i])  # Within 0.5% of level
            
            # Determine 1d trend: swing high > previous swing high = uptrend, swing low < previous swing low = downtrend
            # Use previous completed 1d bar for trend determination
            if i >= 1:
                prev_swing_high = swing_high_1d_aligned[i-1]
                prev_swing_low = swing_low_1d_aligned[i-1]
                curr_swing_high = swing_high_1d_aligned[i]
                curr_swing_low = swing_low_1d_aligned[i]
                
                uptrend = curr_swing_high > prev_swing_high
                downtrend = curr_swing_low < prev_swing_low
            else:
                uptrend = False
                downtrend = False
            
            long_condition = near_618 and uptrend and volume_spike[i]
            short_condition = near_382 and downtrend and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price moves back to 50% level or volume dries up
            fib_50 = swing_low_1d_aligned[i] + 0.5 * diff_1d[i] if not np.isnan(diff_1d[i]) else np.nan
            if not np.isnan(fib_50) and close[i] >= fib_50:
                signals[i] = 0.0
                position = 0
            elif not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price moves back to 50% level or volume dries up
            fib_50 = swing_low_1d_aligned[i] + 0.5 * diff_1d[i] if not np.isnan(diff_1d[i]) else np.nan
            if not np.isnan(fib_50) and close[i] <= fib_50:
                signals[i] = 0.0
                position = 0
            elif not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals