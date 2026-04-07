#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: On 6h timeframe, enter long when price breaks above Camarilla R4 with volume > 1.5x average during uptrend (price above 1d EMA200), enter short when price breaks below Camarilla S4 with volume > 1.5x average during downtrend (price below 1d EMA200). Uses 1d EMA200 trend filter to avoid counter-trend trades. Target: 15-35 trades/year to minimize fee decay while capturing strong momentum moves in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if data not available
        if (np.isnan(vol_ma[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels using previous day's OHLC
        prev_close = close_1d[i-1] if i-1 >= 0 else close_1d[0]
        prev_high = high_1d[i-1] if i-1 >= 0 else high_1d[0]
        prev_low = low_1d[i-1] if i-1 >= 0 else low_1d[0]
        
        # We need 1d OHLC arrays
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d_arr = df_1d['close'].values
        
        # Ensure we have valid data for Camarilla calculation
        if i-1 < 0:
            signals[i] = 0.0
            continue
            
        prev_close = close_1d_arr[i-1]
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        
        # Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Resistance levels
        r3 = prev_close + (range_val * 1.1 / 2)
        r4 = prev_close + (range_val * 1.1)
        # Support levels
        s3 = prev_close - (range_val * 1.1 / 2)
        s4 = prev_close - (range_val * 1.1)
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price closes below R3 or trend changes
            if close[i] < r3 or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 or trend changes
            if close[i] > s3 or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above R4 in uptrend
                if close[i] > r4 and close[i] > ema_200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below S4 in downtrend
                elif close[i] < s4 and close[i] < ema_200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals