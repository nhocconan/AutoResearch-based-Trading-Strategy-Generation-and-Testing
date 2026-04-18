#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot Levels (R1/S1) with 1d volume spike and 1w ADX regime filter.
Long when price breaks above R1 with volume spike in trending market (ADX>25).
Short when price breaks below S1 with volume spike in trending market.
Uses discrete positions (0.0, ±0.25) to minimize churn. Target: 20-40 trades/year.
Works in bull/bear via ADX filter - only trades in clear trends, avoids chop.
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
    atr = np.full(len(high), np.nan)
    dm_plus_smooth = np.full(len(high), np.nan)
    dm_minus_smooth = np.full(len(high), np.nan)
    
    if len(high) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    di_plus = np.full(len(high), np.nan)
    di_minus = np.full(len(high), np.nan)
    dx = np.full(len(high), np.nan)
    
    for i in range(period, len(high)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX
    adx = np.full(len(high), np.nan)
    for i in range(2*period-1, len(high)):
        if i == 2*period-1:
            adx[i] = np.nanmean(dx[period:i+1])
        else:
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    pivot = (high + low + close) / 3
    range_ = high - low
    R1 = close + (range_ * 1.1 / 12)
    S1 = close - (range_ * 1.1 / 12)
    return R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels (R1, S1) on 1d
    R1_1d, S1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate ADX on 1w
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align to 4h timeframe
    R1_1d_4h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_4h = align_htf_to_ltf(prices, df_1d, S1_1d)
    adx_14_1w_4h = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_1d_4h[i]) or np.isnan(S1_1d_4h[i]) or 
            np.isnan(adx_14_1w_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: trending market (ADX > 25)
        trending = adx_14_1w_4h[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 with volume in trending market
            if close[i] > R1_1d_4h[i] and vol_confirmed and trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in trending market
            elif close[i] < S1_1d_4h[i] and vol_confirmed and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 or ADX drops (range market)
            if close[i] < S1_1d_4h[i] or adx_14_1w_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 or ADX drops (range market)
            if close[i] > R1_1d_4h[i] or adx_14_1w_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_ADX"
timeframe = "4h"
leverage = 1.0