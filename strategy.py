#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_volume_v1
Hypothesis: On 4-hour timeframe, use Camarilla pivot levels from 1-day for mean reversion entry in ranging markets.
Long when price touches S3 support with volume confirmation in low volatility regime.
Short when price touches R3 resistance with volume confirmation in low volatility regime.
Exit on opposite touch or volatility expansion.
Designed for 20-40 trades/year to minimize fee drag while capturing mean reversion in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 for no look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    r3 = prev_close + range_ * 1.1 / 2
    s3 = prev_close - range_ * 1.1 / 2
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate ATR for volatility regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 14
    alpha = 1.0 / atr_period
    
    def wilders_smoothing(arr):
        smoothed = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                if i == 0:
                    smoothed[i] = np.nan
                else:
                    smoothed[i] = smoothed[i-1]
            else:
                if i == 0 or np.isnan(smoothed[i-1]):
                    smoothed[i] = arr[i]
                else:
                    smoothed[i] = smoothed[i-1] + alpha * (arr[i] - smoothed[i-1])
        return smoothed
    
    atr = wilders_smoothing(tr)
    
    # Calculate ATR percentile for regime (20-period lookback)
    atr_ma = np.full_like(atr, np.nan)
    for i in range(len(atr)):
        if i < 20:
            atr_ma[i] = np.nan
        else:
            window = atr[max(0, i-19):i+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                atr_ma[i] = np.mean(valid)
    
    atr_ratio = np.full_like(atr, np.nan)
    for i in range(len(atr)):
        if np.isnan(atr[i]) or np.isnan(atr_ma[i]) or atr_ma[i] == 0:
            atr_ratio[i] = np.nan
        else:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    # Volume moving average for confirmation
    vol_ma = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.nan
        else:
            window = volume[max(0, i-19):i+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                vol_ma[i] = np.mean(valid)
    
    volume_ratio = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            volume_ratio[i] = np.nan
        else:
            volume_ratio[i] = volume[i] / vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
            
        # Volatility regime filter: only trade in low volatility (mean reversion works better)
        low_vol = atr_ratio[i] < 1.2
        
        # Volume confirmation: above average volume
        vol_confirm = volume_ratio[i] > 1.1
        
        if position == 1:  # Long position
            # Exit: price touches R3 or volatility expands
            if close[i] >= r3_aligned[i] or not low_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches S3 or volatility expands
            if close[i] <= s3_aligned[i] or not low_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter in low volatility with volume confirmation
            if low_vol and vol_confirm:
                # Long at S3 support
                if close[i] <= s3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short at R3 resistance
                elif close[i] >= r3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals