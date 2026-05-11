#!/usr/bin/env python3
"""
6h_12h_Fibonacci_Extension_Retest_Trend
Hypothesis: Price retesting 126.8% Fibonacci extension levels from 12h swings with trend confirmation (12h EMA20) and volume filter. Works in trending markets by capturing continuation after pullbacks. Uses 12h swing points to project extensions, reducing false signals. Designed for 6h timeframe with 12h trend filter to avoid counter-trend trades. Targets 15-30 trades/year per symbol.
"""

name = "6h_12h_Fibonacci_Extension_Retest_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for swing points and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h close for EMA trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(
        span=20, adjust=False, min_periods=20
    ).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Identify 12h swing highs and lows (simple fractal: higher high/low)
    # Swing high: high > previous high and next high
    # Swing low: low < previous low and next low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    swing_high = np.zeros(len(high_12h), dtype=bool)
    swing_low = np.zeros(len(low_12h), dtype=bool)
    
    for i in range(1, len(high_12h)-1):
        if high_12h[i] > high_12h[i-1] and high_12h[i] > high_12h[i+1]:
            swing_high[i] = True
        if low_12h[i] < low_12h[i-1] and low_12h[i] < low_12h[i+1]:
            swing_low[i] = True
    
    # Calculate Fibonacci extensions from recent swings
    # For uptrend: extension from swing low to swing high
    # For downtrend: extension from swing high to swing low
    ext_1268 = np.full(len(high_12h), np.nan)
    
    last_swing_high_idx = -1
    last_swing_low_idx = -1
    
    for i in range(len(high_12h)):
        if swing_high[i]:
            last_swing_high_idx = i
        if swing_low[i]:
            last_swing_low_idx = i
        
        # Calculate extension when we have both swing points
        if last_swing_high_idx >= 0 and last_swing_low_idx >= 0:
            # Determine trend based on order of swings
            if last_swing_low_idx < last_swing_high_idx:
                # Uptrend: low -> high, extension above high
                swing_low_price = low_12h[last_swing_low_idx]
                swing_high_price = high_12h[last_swing_high_idx]
                diff = swing_high_price - swing_low_price
                ext_1268[i] = swing_high_price + 1.268 * diff
            else:
                # Downtrend: high -> low, extension below low
                swing_high_price = high_12h[last_swing_high_idx]
                swing_low_price = low_12h[last_swing_low_idx]
                diff = swing_high_price - swing_low_price
                ext_1268[i] = swing_low_price - 1.268 * diff
    
    # Align extension level to 6h timeframe
    ext_1268_aligned = align_htf_to_ltf(prices, df_12h, ext_1268)
    
    # Volume confirmation (24-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ext_1268_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_filter = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price near 126.8% extension (retest) + above 12h EMA20 + volume
            # Allow 0.5% tolerance for retest
            ext_level = ext_1268_aligned[i]
            if (not np.isnan(ext_level) and
                abs(close[i] - ext_level) / ext_level < 0.005 and  # within 0.5%
                close[i] > ema_20_12h_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price near 126.8% extension (retest) + below 12h EMA20 + volume
            elif (not np.isnan(ext_level) and
                  abs(close[i] - ext_level) / ext_level < 0.005 and  # within 0.5%
                  close[i] < ema_20_12h_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price moves 1% away from extension level or trend reversal
            ext_level = ext_1268_aligned[i]
            if not np.isnan(ext_level):
                if position == 1:
                    # Exit long: price drops below extension or trend turns down
                    if (close[i] < ext_level * 0.99) or \
                       (close[i] < ema_20_12h_aligned[i]):
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                elif position == -1:
                    # Exit short: price rises above extension or trend turns up
                    if (close[i] > ext_level * 1.01) or \
                       (close[i] > ema_20_12h_aligned[i]):
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
            else:
                # No extension level, exit on trend reversal
                if position == 1 and close[i] < ema_20_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > ema_20_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
    
    return signals