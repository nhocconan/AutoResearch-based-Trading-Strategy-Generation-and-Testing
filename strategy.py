#!/usr/bin/env python3
"""
4h_MultiTimeframe_Camarilla_Pivot_Breakout_V1
Hypothesis: Use daily Camarilla pivot levels (S1/R1) for breakout signals, with 1d volume confirmation and 1w ADX trend filter.
Go long when price breaks above R1 with volume > 1.5x 20-day average and weekly ADX > 25.
Go short when price breaks below S1 with volume > 1.5x 20-day average and weekly ADX > 25.
Exit when price returns to the pivot point (PP) or reverses with opposite volume spike.
Designed for low-frequency, high-conviction trades (target: 20-40/year) to avoid fee drag.
Works in bull markets via R1 breakouts and in bear via S1 breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        pp_1d[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1_1d[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
        s1_1d[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ADX(14)
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = np.full_like(high, np.nan)
        dm_plus_smooth = np.full_like(high, np.nan)
        dm_minus_smooth = np.full_like(high, np.nan)
        
        # Initial average
        if n >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
            dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
        
        # Wilder smoothing
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.full_like(high, np.nan)
        di_minus = np.full_like(high, np.nan)
        dx = np.full_like(high, np.nan)
        
        for i in range(period-1, n):
            if atr[i] != 0:
                di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
                di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
                if (di_plus[i] + di_minus[i]) != 0:
                    dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
        
        # ADX
        adx = np.full_like(high, np.nan)
        if n >= 2 * period - 1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, n):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    vol_period = 20
    
    if len(volume_1d) >= vol_period:
        for i in range(vol_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i - vol_period:i])
    
    # Align all indicators to 4h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, vol_period) + 1  # Need at least one day of pivot data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Trend filter: weekly ADX > 25
        trend_filter = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 with volume and trend
            if close[i] > r1_1d_aligned[i] and vol_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and trend
            elif close[i] < s1_1d_aligned[i] and vol_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to PP or breaks below S1
            if close[i] <= pp_1d_aligned[i] or close[i] < s1_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to PP or breaks above R1
            if close[i] >= pp_1d_aligned[i] or close[i] > r1_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_MultiTimeframe_Camarilla_Pivot_Breakout_V1"
timeframe = "4h"
leverage = 1.0