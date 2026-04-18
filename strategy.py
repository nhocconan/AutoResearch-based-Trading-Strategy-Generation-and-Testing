#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot reversal with volume spike and 1d ADX trend filter.
Long when price rejects S1 with volume in uptrend (ADX>25); short when price rejects R1 with volume in downtrend.
Uses 1d Camarilla levels for institutional reference points. Designed for 20-40 trades/year to minimize fee drag.
Works in bull/bear via ADX trend filter - only trades with institutional level rejection in trending markets.
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
        # Initial values
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        # Wilder smoothing
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Directional Indicators
    plus_di = np.full(len(dm_plus), np.nan)
    minus_di = np.full(len(dm_minus), np.nan)
    dx = np.full(len(dm_plus), np.nan)
    
    for i in range(period, len(tr)):
        if atr[i] != 0:
            plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
            minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX
    adx = np.full(len(dx), np.nan)
    for i in range(2*period, len(dx)):
        if np.isnan(dx[i-1]):
            adx[i] = np.nanmean(dx[period:i+1])
        else:
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    if len(high) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]), 
                np.array([]), np.array([]), np.array([]), np.array([]))
    
    pivot = (high + low + close) / 3
    range_val = high - low
    
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    
    return R1, R2, R3, R4, S1, S2, S3, S4

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
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(
        high_1d, low_1d, close_1d)
    
    # Align to 4h timeframe
    adx_14_1d_4h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    R1_1d_4h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_4h = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_1d_4h[i]) or np.isnan(R1_1d_4h[i]) or 
            np.isnan(S1_1d_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_14_1d_4h[i] > 25
        
        if position == 0:
            # Long: price at or below S1 with volume in uptrend
            if close[i] <= S1_1d_4h[i] and vol_confirmed and trending:
                signals[i] = 0.25
                position = 1
            # Short: price at or above R1 with volume in downtrend
            elif close[i] >= R1_1d_4h[i] and vol_confirmed and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses above S1 or loses volume/trend
            if close[i] > S1_1d_4h[i] or not vol_confirmed or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below R1 or loses volume/trend
            if close[i] < R1_1d_4h[i] or not vol_confirmed or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_S1R1_Volume_ADX"
timeframe = "4h"
leverage = 1.0