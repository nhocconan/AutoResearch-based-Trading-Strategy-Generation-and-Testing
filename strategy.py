#!/usr/bin/env python3
"""
6h Camarilla Pivot with 1d EMA Trend and Volume Confirmation
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
filtered by daily trend and volume spikes provide high-probability entries.
Works in bull markets via breakouts at R4/S4 and bear markets via mean reversion 
at R3/S3. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v2"
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
    
    # 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    hl_range = high_1d - low_1d
    r4 = close_1d + 1.5 * hl_range
    r3 = close_1d + 1.0 * hl_range
    s3 = close_1d - 1.0 * hl_range
    s4 = close_1d - 1.5 * hl_range
    
    # Align pivots to 6s timeframe (using previous day's values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA(20) for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume filter: current volume > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion failure) or 
            # price crosses above R4 with momentum (take profit)
            if (close[i] < s3_aligned[i] or 
                close[i] > r4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion failure) or 
            # price crosses below S4 with momentum (take profit)
            if (close[i] > r3_aligned[i] or 
                close[i] < s4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            uptrend = close[i] > ema_20_1d_aligned[i]
            downtrend = close[i] < ema_20_1d_aligned[i]
            
            # Mean reversion longs at S3 (price < S3 and reversing up)
            # Mean reversion shorts at R3 (price > R3 and reversing down)
            if i > 0:
                # Long: price crosses above S3 with downtrend (fade) and volume spike
                if (close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i-1] and
                    downtrend and vol_spike[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price crosses below R3 with uptrend (fade) and volume spike
                elif (close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i-1] and
                      uptrend and vol_spike[i]):
                    position = -1
                    signals[i] = -0.25
                # Breakout longs: price breaks above R4 with uptrend and volume spike
                elif (close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i-1] and
                      uptrend and vol_spike[i]):
                    position = 1
                    signals[i] = 0.25
                # Breakout shorts: price breaks below S4 with downtrend and volume spike
                elif (close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i-1] and
                      downtrend and vol_spike[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals