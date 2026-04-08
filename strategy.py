#!/usr/bin/env python3
"""
6h_1d_Camarilla_Pivot_Volume_Breakout
Hypothesis: Camarilla pivot levels from 1-day chart provide strong intraday support/resistance.
- Breakout above R4 or below S4 with volume continuation indicates institutional participation.
- Fade at R3/S3 with volume exhaustion for mean reversion in range-bound markets.
- Works in both bull/bear markets as it adapts to volatility and volume confirmation.
- Uses 1-day Camarilla levels as they are widely watched by institutions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_Pivot_Volume_Breakout"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    pivot = (high + low + close) / 3
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    r2 = close + range_val * 1.1 / 6
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels (updated daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Arrays to store Camarilla levels
    r4_1d = np.full_like(close_1d, np.nan)
    r3_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    
    # Calculate for each day (using previous day's data to avoid look-ahead)
    for i in range(1, len(close_1d)):
        _, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
        r4_1d[i] = r4
        r3_1d[i] = r3
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align 1d Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if Camarilla data not available
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below R3 (mean reversion) OR stop loss logic via signal=0
            if close[i] < r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price crosses above S3 (mean reversion)
            if close[i] > s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long breakout: price breaks above R4 with volume
            if close[i] > r4_1d_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short breakout: price breaks below S4 with volume
            elif close[i] < s4_1d_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
            # Long mean reversion: price touches S3 with volume exhaustion (contrarian)
            elif close[i] <= s3_1d_aligned[i] and not volume_filter[i]:
                # Only take mean reversion if volume is below average (exhaustion)
                position = 1
                signals[i] = 0.25
            # Short mean reversion: price touches R3 with volume exhaustion
            elif close[i] >= r3_1d_aligned[i] and not volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals