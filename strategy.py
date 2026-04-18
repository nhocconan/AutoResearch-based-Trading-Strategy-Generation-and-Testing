#!/usr/bin/env python3
"""
12h_1d_Long_Range_MeanReversion
Hypothesis: In a ranging market (12h), price tends to revert to the mean of the prior day's range. 
Enter long when price touches or dips below the 1d low and closes back above it on 12h, 
with volume confirmation and ADX < 25 (non-trending). Exit when price reaches the 1d high or ADX rises above 25.
Works in both bull and bear markets because it exploits mean reversion in ranging conditions, 
which occur regardless of trend direction. Uses 1d range as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for range and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smooth TR, DM+
        tr_period = np.full_like(tr, np.nan)
        dm_plus_period = np.full_like(dm_plus, np.nan)
        dm_minus_period = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            tr_period[period-1] = np.nansum(tr[1:period+1])
            dm_plus_period[period-1] = np.nansum(dm_plus[1:period+1])
            dm_minus_period[period-1] = np.nansum(dm_minus[1:period+1])
            
            for i in range(period, len(tr)):
                tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
                dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / period) + dm_plus[i]
                dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / period) + dm_minus[i]
        
        # DI+ and DI-
        di_plus = np.full_like(tr, np.nan)
        di_minus = np.full_like(tr, np.nan)
        valid = tr_period != 0
        di_plus[valid] = 100 * dm_plus_period[valid] / tr_period[valid]
        di_minus[valid] = 100 * dm_minus_period[valid] / tr_period[valid]
        
        # DX and ADX
        dx = np.full_like(tr, np.nan)
        di_sum = di_plus + di_minus
        valid_dx = di_sum != 0
        dx[valid_dx] = 100 * np.abs(di_plus[valid_dx] - di_minus[valid_dx]) / di_sum[valid_dx]
        
        adx = np.full_like(tr, np.nan)
        if len(dx) >= 2*period - 1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d range: high and low
    range_high_1d = high_1d
    range_low_1d = low_1d
    range_high_aligned = align_htf_to_ltf(prices, df_1d, range_high_1d)
    range_low_aligned = align_htf_to_ltf(prices, df_1d, range_low_1d)
    
    # Volume confirmation: volume > 1.5x 24-period average (on 12h)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = max(30, vol_period)  # Need sufficient data
    
    for i in range(start_idx, n):
        # Skip if any data is not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(range_high_aligned[i]) or 
            np.isnan(range_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: only trade when ADX < 25 (ranging market)
        ranging = adx_1d_aligned[i] < 25
        
        if position == 0:
            # Long when price touches or goes below 1d low and closes back above it
            if low[i] <= range_low_aligned[i] and close[i] > range_low_aligned[i] and vol_confirm and ranging:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit when price reaches 1d high or ADX rises above 25 (trending)
            if high[i] >= range_high_aligned[i] or adx_1d_aligned[i] >= 25:
                signals[i] = 0.0  # flat
                position = 0
            else:
                signals[i] = 0.25  # remain long
    
    return signals

name = "12h_1d_Long_Range_MeanReversion"
timeframe = "12h"
leverage = 1.0