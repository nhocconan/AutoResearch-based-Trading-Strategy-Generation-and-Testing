#!/usr/bin/env python3
"""
12h Camarilla Pivot R1/S1 Breakout with Volume Confirmation and Weekly ADX Trend Filter.
Long when price breaks above R1 with volume and weekly ADX > 25 (trending up).
Short when price breaks below S1 with volume and weekly ADX > 25 (trending down).
Uses weekly ADX to avoid whipsaws in ranging markets. Designed for 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    
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
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr = np.full(len(tr), np.nan)
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    
    if len(tr) >= period:
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # DI+ and DI-
    di_plus = np.full(len(dm_plus), np.nan)
    di_minus = np.full(len(dm_minus), np.nan)
    for i in range(period, len(tr)):
        if atr[i] != 0:
            di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
    
    # DX and ADX
    dx = np.full(len(di_plus), np.nan)
    for i in range(period, len(tr)):
        if (di_plus[i] + di_minus[i]) != 0:
            dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    adx = np.full(len(dx), np.nan)
    if len(tr) >= 2*period:
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(tr)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX(14)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly ADX
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Calculate daily Camarilla pivots
    _, r1_1d, s1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 12h timeframe
    adx_14_1w_12h = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    r1_1d_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_1w_12h[i]) or np.isnan(r1_1d_12h[i]) or 
            np.isnan(s1_1d_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and weekly ADX > 25
            if close[i] > r1_1d_12h[i] and adx_14_1w_12h[i] > 25 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and weekly ADX > 25
            elif close[i] < s1_1d_12h[i] and adx_14_1w_12h[i] > 25 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 or ADX weakens (< 20)
            if close[i] < s1_1d_12h[i] or adx_14_1w_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 or ADX weakens (< 20)
            if close[i] > r1_1d_12h[i] or adx_14_1w_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0