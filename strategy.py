#!/usr/bin/env python3
"""
6h ADX + Williams Alligator combination
Hypothesis: ADX > 25 filters trending markets, while Williams Alligator (SMAs of 13, 8, 5) confirms direction. In strong trends (ADX > 25), we go long when green line > red line > blue line (bullish alignment) and short when red > green > blue (bearish alignment). Uses 1d timeframe for ADX to avoid whipsaw, with 6h for entry/exit. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_alligator_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d (14-period)
    period = 14
    n_1d = len(high_1d)
    tr_1d = np.zeros(n_1d)
    plus_dm_1d = np.zeros(n_1d)
    minus_dm_1d = np.zeros(n_1d)
    
    for i in range(1, n_1d):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm_1d[i] = high_diff
        else:
            plus_dm_1d[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm_1d[i] = low_diff
        else:
            minus_dm_1d[i] = 0
    
    # Smoothed values
    atr_1d = np.zeros(n_1d)
    plus_di_1d = np.zeros(n_1d)
    minus_di_1d = np.zeros(n_1d)
    
    if n_1d >= period:
        atr_1d[period-1] = np.mean(tr_1d[1:period])
        plus_di_1d[period-1] = np.mean(plus_dm_1d[1:period]) / atr_1d[period-1] * 100 if atr_1d[period-1] != 0 else 0
        minus_di_1d[period-1] = np.mean(minus_dm_1d[1:period]) / atr_1d[period-1] * 100 if atr_1d[period-1] != 0 else 0
        
        for i in range(period, n_1d):
            atr_1d[i] = (atr_1d[i-1] * (period-1) + tr_1d[i]) / period
            plus_di_1d[i] = (plus_di_1d[i-1] * (period-1) + plus_dm_1d[i]) / atr_1d[i] * 100 if atr_1d[i] != 0 else 0
            minus_di_1d[i] = (minus_di_1d[i-1] * (period-1) + minus_dm_1d[i]) / atr_1d[i] * 100 if atr_1d[i] != 0 else 0
    
    # Calculate DX and ADX
    dx_1d = np.zeros(n_1d)
    adx_1d = np.zeros(n_1d)
    
    for i in range(period, n_1d):
        di_sum = plus_di_1d[i] + minus_di_1d[i]
        if di_sum != 0:
            dx_1d[i] = abs(plus_di_1d[i] - minus_di_1d[i]) / di_sum * 100
        else:
            dx_1d[i] = 0
    
    if n_1d >= 2 * period - 1:
        adx_1d[2*period-2] = np.mean(dx_1d[period-1:2*period-1])
        for i in range(2*period-1, n_1d):
            adx_1d[i] = (adx_1d[i-1] * (period-1) + dx_1d[i]) / period
    
    # Williams Alligator on 1d (SMAs of median price)
    median_price_1d = (high_1d + low_1d) / 2
    jaw_1d = np.full(n_1d, np.nan)  # 13-period SMA
    teeth_1d = np.full(n_1d, np.nan)  # 8-period SMA
    lips_1d = np.full(n_1d, np.nan)  # 5-period SMA
    
    for i in range(12, n_1d):  # 13-period
        jaw_1d[i] = np.mean(median_price_1d[i-12:i+1])
    for i in range(7, n_1d):  # 8-period
        teeth_1d[i] = np.mean(median_price_1d[i-7:i+1])
    for i in range(4, n_1d):  # 5-period
        lips_1d[i] = np.mean(median_price_1d[i-4:i+1])
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 6-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 6:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[5] = np.mean(tr[:5])
            for i in range(6, n):
                atr[i] = (atr[i-1] * 5 + tr[i-1]) / 6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # Need enough data for indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(jaw_1d_aligned[i]) or 
            np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Alligator lines not bullish aligned OR stoploss
            bullish_aligned = (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i])
            if (not bullish_aligned or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: Alligator lines not bearish aligned OR stoploss
            bearish_aligned = (jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i])
            if (not bearish_aligned or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Strong trend filter: ADX > 25
                strong_trend = adx_1d_aligned[i] > 25
                
                # Alligator alignment
                bullish_aligned = (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i])
                bearish_aligned = (jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i])
                
                # Long: strong trend + bullish alignment
                if strong_trend and bullish_aligned:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: strong trend + bearish alignment
                elif strong_trend and bearish_aligned:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals