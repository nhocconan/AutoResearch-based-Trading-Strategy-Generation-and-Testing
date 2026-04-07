#!/usr/bin/env python3
"""
6h_adx_di_crossover_1w_trend_v1
Hypothesis: On 6-hour timeframe, use weekly ADX and DI crossover to identify strong trends, 
entering long when +DI crosses above -DI with ADX > 25, short when -DI crosses above +DI with ADX > 25.
Exit when ADX falls below 20 (trend weakening) or opposite DI crossover occurs.
Weekly trend filter ensures alignment with major trend, reducing whipsaw in ranging markets.
ADX > 25 ensures we only trade strong trends, avoiding chop.
Target: 20-30 trades/year to minimize fee decay while capturing sustained trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_di_crossover_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # Calculate ADX and DI on weekly
    period = 14
    
    # True Range
    tr1 = wh[1:] - wl[1:]
    tr2 = np.abs(wh[1:] - wc[:-1])
    tr3 = np.abs(wl[1:] - wc[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((wh[1:] - wh[:-1]) > (wl[:-1] - wl[1:]), np.maximum(wh[1:] - wh[:-1], 0), 0)
    dm_minus = np.where((wl[:-1] - wl[1:]) > (wh[1:] - wh[:-1]), np.maximum(wl[:-1] - wl[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        """Wilder smoothing (same as RSI)"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])  # skip first NaN
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, period)
    dm_plus_smooth = smooth_wilder(dm_plus, period)
    dm_minus_smooth = smooth_wilder(dm_minus, period)
    
    # DI+
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    # DI-
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX (smoothed DX)
    adx = smooth_wilder(dx, period)
    
    # Align to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_1w, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_1w, di_minus)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to avoid NaN issues
        # Skip if ADX not available
        if np.isnan(adx_aligned[i]) or np.isnan(di_plus_aligned[i]) or np.isnan(di_minus_aligned[i]):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        di_plus_val = di_plus_aligned[i]
        di_minus_val = di_minus_aligned[i]
        
        # Previous values for crossover detection
        if i > 1:
            di_plus_prev = di_plus_aligned[i-1]
            di_minus_prev = di_minus_aligned[i-1]
        else:
            di_plus_prev = di_plus_val
            di_minus_prev = di_minus_val
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when ADX falls below 20 (trend weakening)
            if adx_val < 20:
                exit_long = True
            # Exit when -DI crosses above +DI
            elif di_minus_prev <= di_plus_prev and di_minus_val > di_plus_val:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when ADX falls below 20 (trend weakening)
            if adx_val < 20:
                exit_short = True
            # Exit when +DI crosses above -DI
            elif di_plus_prev <= di_minus_prev and di_plus_val > di_minus_val:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: ADX > 25 AND +DI crosses above -DI
            long_entry = (adx_val > 25) and (di_plus_prev <= di_minus_prev) and (di_plus_val > di_minus_val)
            
            # Short entry: ADX > 25 AND -DI crosses above +DI
            short_entry = (adx_val > 25) and (di_minus_prev <= di_plus_prev) and (di_minus_val > di_plus_val)
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals