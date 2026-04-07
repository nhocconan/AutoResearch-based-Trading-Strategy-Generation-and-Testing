#!/usr/bin/env python3
"""
6h Camarilla Pivot + 1d EMA Trend + Volume Confirmation
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
identify institutional support/resistance. In 6h timeframe, we fade at R3/S3 when 
price rejects extreme levels, and breakout continuation at R4/S4 when price 
breaks with volume. 1d EMA(50) filters for higher timeframe trend alignment. 
Volume > 1.5x average confirms institutional participation. Works in bull/bear 
by adapting to pivot structure - mean reversion in ranges, breakout in trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla Pivot levels from previous day
    # Typical Price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Use previous day's values for today's pivot (avoid look-ahead)
    prev_typical = np.roll(typical_price, 1)
    prev_typical[0] = np.nan  # First value has no previous
    
    # Calculate pivot and ranges
    pivot = prev_typical
    range_val = high - low
    prev_range = np.roll(range_val, 1)
    prev_range[0] = np.nan
    
    # Camarilla levels (based on previous day)
    r4 = pivot + (prev_range * 1.1 / 2)
    r3 = pivot + (prev_range * 1.1 / 4)
    s3 = pivot - (prev_range * 1.1 / 4)
    s4 = pivot - (prev_range * 1.1 / 2)
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or 
            np.isnan(s4[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 (mean reversion fail) OR 
            # price breaks above R4 and fails (bull trap) OR trend reverses
            if (close[i] < s3[i] or 
                (close[i] > r4[i] and close[i] < r4[i-1]) or  # Failed breakout
                close[i] < ema_50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (mean reversion fail) OR
            # price breaks below S4 and fails (bear trap) OR trend reverses
            if (close[i] > r3[i] or 
                (close[i] < s4[i] and close[i] > s4[i-1]) or  # Failed breakout
                close[i] > ema_50_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long mean reversion: price rejects S3 with volume
            if (close[i] <= s3[i] and 
                close[i] > s3[i-1] and  # Rejection bounce
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short mean reversion: price rejects R3 with volume
            elif (close[i] >= r3[i] and 
                  close[i] < r3[i-1] and  # Rejection rejection
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
            # Long breakout: price breaks R4 with volume and trend alignment
            elif (close[i] > r4[i] and 
                  close[i] > ema_50_aligned[i] and 
                  vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakout: price breaks S4 with volume and trend alignment
            elif (close[i] < s4[i] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals