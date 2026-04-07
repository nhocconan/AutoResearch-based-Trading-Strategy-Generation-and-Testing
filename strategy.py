#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Pivot Breakout with Volume Confirmation
# Hypothesis: Weekly pivot levels act as strong support/resistance. Breakouts with volume
# confirmation capture momentum in both bull and bear markets. Uses 1d for weekly pivot
# calculation (more reliable than direct weekly data due to potential gaps).
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

name = "6h_weekly_pivot_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data
    # Using last 5 days to approximate weekly OHLC
    high_5d = np.max(high[-5:]) if len(high) >= 5 else np.max(high)
    low_5d = np.min(low[-5:]) if len(low) >= 5 else np.min(low)
    close_5d = close[-1]
    
    # Weekly pivot calculation
    pp = (high_5d + low_5d + close_5d) / 3.0
    r1 = 2 * pp - low_5d
    s1 = 2 * pp - high_5d
    r2 = pp + (high_5d - low_5d)
    s2 = pp - (high_5d - low_5d)
    r3 = high_5d + 2 * (pp - low_5d)
    s3 = low_5d - 2 * (high_5d - pp)
    
    # Volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Recalculate weekly pivots for current point (using available data)
        if i >= 5:
            lookback_high = np.max(high[i-4:i+1])
            lookback_low = np.min(low[i-4:i+1])
            lookback_close = close[i]
            
            pp = (lookback_high + lookback_low + lookback_close) / 3.0
            r1 = 2 * pp - lookback_low
            s1 = 2 * pp - lookback_high
            r2 = pp + (lookback_high - lookback_low)
            s2 = pp - (lookback_high - lookback_low)
            r3 = lookback_high + 2 * (pp - lookback_low)
            s3 = lookback_low - 2 * (lookback_high - pp)
        else:
            # Not enough data, skip
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price breaks below S1 or volume fails
            if close[i] < s1 or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R1 or volume fails
            if close[i] > r1 or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout above R1 with volume
            if close[i] > r1 and vol_ok:
                position = 1
                signals[i] = 0.25
            # Short breakdown below S1 with volume
            elif close[i] < s1 and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals