#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot Breakout with Volume Confirmation and 1d ADX Trend Filter.
Long when price breaks above R1 with volume and ADX>25; short when breaks below S1 with volume and ADX>25.
Uses 1d Camarilla levels for structure and 1d ADX to filter ranging markets. Designed for 20-40 trades/year.
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
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    atr = np.full(len(tr), np.nan)
    plus_di = np.full(len(tr), np.nan)
    minus_di = np.full(len(tr), np.nan)
    
    if len(tr) >= period:
        # Initial averages
        atr[period] = np.nanmean(tr[1:period+1])
        plus_dm_sum = np.nansum(plus_dm[1:period+1])
        minus_dm_sum = np.nansum(minus_dm[1:period+1])
        
        if not np.isnan(atr[period]) and atr[period] != 0:
            plus_di[period] = 100 * plus_dm_sum / (atr[period] * period)
            minus_di[period] = 100 * minus_dm_sum / (atr[period] * period)
        
        # Wilder smoothing
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            plus_dm_val = plus_dm[i]
            minus_dm_val = minus_dm[i]
            plus_di[i] = (plus_di[i-1] * (period - 1) + plus_dm_val) / period if not np.isnan(atr[i]) and atr[i] != 0 else np.nan
            minus_di[i] = (minus_di[i-1] * (period - 1) + minus_dm_val) / period if not np.isnan(atr[i]) and atr[i] != 0 else np.nan
    
    # DX and ADX
    dx = np.full(len(tr), np.nan)
    adx = np.full(len(tr), np.nan)
    
    for i in range(period, len(tr)):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    if len(tr) >= 2 * period:
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(tr)):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day."""
    if len(high) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Typical price
    typical = (high + low + close) / 3
    
    # Pivot point
    pivot = typical
    
    # Range
    range_val = high - low
    
    # Camarilla levels
    r1 = close + (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    s1 = close - (range_val * 1.1 / 12)
    s2 = close - (range_val * 1.1 / 6)
    
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate Camarilla levels on 1d
    _, r1_1d, _, s1_1d, _ = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 4h timeframe
    adx_14_1d_4h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    r1_1d_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_1d_4h[i]) or np.isnan(r1_1d_4h[i]) or 
            np.isnan(s1_1d_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # ADX trend filter: only trade when trending (ADX > 25)
        trending = adx_14_1d_4h[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 with volume and trend
            if close[i] > r1_1d_4h[i] and vol_confirmed and trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and trend
            elif close[i] < s1_1d_4h[i] and vol_confirmed and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below R1 or ADX weakens
            if close[i] < r1_1d_4h[i] or adx_14_1d_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above S1 or ADX weakens
            if close[i] > s1_1d_4h[i] or adx_14_1d_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_ADX"
timeframe = "4h"
leverage = 1.0