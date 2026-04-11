#!/usr/bin/env python3
# 6h_1d_ellis_618_v1
# Strategy: 6h Fibonacci retracement (61.8%) from daily swing high/low + volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Daily swing high/low defines major support/resistance. Price retracing to 61.8%
# of the daily range with volume confirmation indicates institutional interest and continuation.
# Works in bull markets via long at 61.8% of daily pullback, bear markets via short at 61.8% of daily bounce.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ellis_618_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Daily swing high/low (10-period lookback for swing points)
    # Swing high: highest high in last 10 days
    # Swing low: lowest low in last 10 days
    lookback = 10
    if len(df_1d) < lookback:
        return np.zeros(n)
    
    # Calculate swing points with proper lookback
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Rolling max/min for swing points
    swing_high = pd.Series(daily_high).rolling(window=lookback, min_periods=lookback).max().values
    swing_low = pd.Series(daily_low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 61.8% retracement level: swing_low + 0.618 * (swing_high - swing_low)
    daily_range = swing_high - swing_low
    fib_618 = swing_low + 0.618 * daily_range
    
    # Align Fibonacci level from daily to 6h
    fib_618_6h = align_htf_to_ltf(prices, df_1d, fib_618)
    
    # 6h EMA20 for trend filter (avoid counter-trend)
    ema_20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 6h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(fib_618_6h[i]) or np.isnan(ema_20_6h[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Determine if we are in a daily uptrend or downtrend based on swing points
        # Uptrend: today's swing high > yesterday's swing high AND today's swing low > yesterday's swing low
        # Downtrend: today's swing high < yesterday's swing high AND today's swing low < yesterday's swing low
        if i >= len(df_1d) * 4:  # Approximate 6h bars per day (varies, but safe)
            daily_idx = i // 4  # Rough daily index from 6h
            if daily_idx >= 1 and daily_idx < len(df_1d):
                prev_swing_high = swing_high[daily_idx-1]
                prev_swing_low = swing_low[daily_idx-1]
                curr_swing_high = swing_high[daily_idx]
                curr_swing_low = swing_low[daily_idx]
                
                daily_uptrend = (curr_swing_high > prev_swing_high) and (curr_swing_low > prev_swing_low)
                daily_downtrend = (curr_swing_high < prev_swing_high) and (curr_swing_low < prev_swing_low)
            else:
                daily_uptrend = False
                daily_downtrend = False
        else:
            daily_uptrend = False
            daily_downtrend = False
        
        # Entry conditions
        # Long: Price near 61.8% fib level (within 0.5%) AND daily uptrend AND volume confirmation
        fib_proximity_long = abs(close[i] - fib_618_6h[i]) / close[i] < 0.005
        if fib_proximity_long and daily_uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price near 61.8% fib level (within 0.5%) AND daily downtrend AND volume confirmation
        elif fib_proximity_long and daily_downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price moves back to swing level (invalidates the retracement premise)
        elif position == 1 and close[i] > swing_high[min(i//4, len(swing_high)-1)]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < swing_low[min(i//4, len(swing_low)-1)]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals