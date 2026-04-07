#!/usr/bin/env python3
"""
6H Camarilla Pivot + Daily EMA + Volume Confirmation
Hypothesis: Camarilla pivot levels from daily timeframe identify key support/resistance. 
Price rejection at R3/S3 levels with EMA(21) trend alignment and volume confirmation 
provides high-probability mean-reversion entries. Works in ranging markets (2022-2024) 
and captures reversals in trending markets (2021, 2025). Target: 15-25 trades/year per symbol.
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
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # where C, H, L are from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First value will be invalid (rolled from end), but we'll handle with isnan check
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    r3 = prev_close + (rang * 1.1 / 4)
    s3 = prev_close - (rang * 1.1 / 4)
    r4 = prev_close + (rang * 1.1 / 2)
    s4 = prev_close - (rang * 1.1 / 2)
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA(21) for trend filter
    ema_21 = close_1d.ewm(span=21, adjust=False).mean().values
    ema_21_6h = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Volume filter (>1.5x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(ema_21_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (target) or breaks below S4 (stop)
            if close[i] <= s3_6h[i] or close[i] < s4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R3 (target) or breaks above R4 (stop)
            if close[i] >= r3_6h[i] or close[i] > r4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion long at S3 with trend alignment
            if (close[i] <= s3_6h[i] and 
                close[i] > ema_21_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion short at R3 with trend alignment
            elif (close[i] >= r3_6h[i] and 
                  close[i] < ema_21_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals