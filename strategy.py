#!/usr/bin/env python3
"""
6H Camarilla Pivot + Daily EMA + Volume Confirmation
Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) from daily timeframe act as strong support/resistance.
Breakouts above R4 or below S4 with daily EMA trend alignment and volume confirmation capture strong momentum.
Mean reversion at R3/S3 with trend filter provides additional entries. Designed for 6h timeframe to balance
trade frequency and signal quality in both bull and bear markets. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_daily_ema_volume_v1"
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
    
    # Daily data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla levels: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # Using previous day's values
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Calculate pivot-based levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = prev_close + range_hl * 1.1 / 2
    r3 = prev_close + range_hl * 1.1 / 4
    s3 = prev_close - range_hl * 1.1 / 4
    s4 = prev_close - range_hl * 1.1 / 2
    
    # Align to 6s timeframe (previous day's levels are known at open)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4.values)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Daily EMA(21) for trend filter
    ema_21 = df_1d['close'].ewm(span=21, adjust=False).mean().values
    ema_21_6h = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Volume filter (>1.5x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(ema_21_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (mean reversion fail) or trend reverses
            if close[i] < s3_6h[i] or close[i] < ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (mean reversion fail) or trend reverses
            if close[i] > r3_6h[i] or close[i] > ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long at R4 with trend alignment
            if (close[i] >= r4_6h[i] and 
                close[i] > ema_21_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short at S4 with trend alignment
            elif (close[i] <= s4_6h[i] and 
                  close[i] < ema_21_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
            # Mean reversion long at S3 with trend alignment
            elif (close[i] <= s3_6h[i] and 
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