#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot reversal with 1d volume confirmation and ADX trend filter.
- Long: price touches S1, rebounds above S1, ADX > 20 (trending), volume > 1.3x average
- Short: price touches R1, rejects below R1, ADX > 20, volume > 1.3x average
- Exit: opposite touch (R4 for long, S4 for short) or ADX < 15
- Uses Camarilla from prior 1d for structure, avoiding whipsaws in ranging markets.
Designed for 12-37 trades/year (50-150 total) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index."""
    if len(high) < period * 2:
        return np.full(len(high), np.nan)
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM
    atr = np.full(len(tr), np.nan)
    if len(tr) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    if len(dm_plus) >= period:
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
        for i in range(period, len(dm_plus)):
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Calculate Directional Indicators
    plus_di = np.full(len(dm_plus), np.nan)
    minus_di = np.full(len(dm_minus), np.nan)
    for i in range(period, len(atr)):
        if atr[i] != 0:
            plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
            minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.full(len(plus_di), np.nan)
    for i in range(period, len(plus_di)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.full(len(dx), np.nan)
    if len(dx) >= 2 * period - 1:
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d (H, L, C)
    camarilla_H4 = np.full(len(high_1d), np.nan)
    camarilla_H3 = np.full(len(high_1d), np.nan)
    camarilla_H2 = np.full(len(high_1d), np.nan)
    camarilla_H1 = np.full(len(high_1d), np.nan)
    camarilla_L1 = np.full(len(low_1d), np.nan)
    camarilla_L2 = np.full(len(low_1d), np.nan)
    camarilla_L3 = np.full(len(low_1d), np.nan)
    camarilla_L4 = np.full(len(low_1d), np.nan)
    
    for i in range(1, len(high_1d)):  # Start from 1 to use previous day
        # Use previous day's H, L, C
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        range_val = H - L
        
        camarilla_H4[i] = C + range_val * 1.5
        camarilla_H3[i] = C + range_val * 1.25
        camarilla_H2[i] = C + range_val * 1.166
        camarilla_H1[i] = C + range_val * 1.083
        camarilla_L1[i] = C - range_val * 1.083
        camarilla_L2[i] = C - range_val * 1.166
        camarilla_L3[i] = C - range_val * 1.25
        camarilla_L4[i] = C - range_val * 1.5
    
    # Calculate ADX (14-period) on 1d
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align to 12h timeframe
    camarilla_H4_12h = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_H3_12h = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_H2_12h = align_htf_to_ltf(prices, df_1d, camarilla_H2)
    camarilla_H1_12h = align_htf_to_ltf(prices, df_1d, camarilla_H1)
    camarilla_L1_12h = align_htf_to_ltf(prices, df_1d, camarilla_L1)
    camarilla_L2_12h = align_htf_to_ltf(prices, df_1d, camarilla_L2)
    camarilla_L3_12h = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_L4_12h = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    adx_14_1d_12h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need Camarilla, ADX, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_H1_12h[i]) or np.isnan(camarilla_L1_12h[i]) or 
            np.isnan(adx_14_1d_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: price touches S1 and rebounds above it, ADX > 20, volume confirmation
            if low[i] <= camarilla_L1_12h[i] and close[i] > camarilla_L1_12h[i] and adx_14_1d_12h[i] > 20 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 and rejects below it, ADX > 20, volume confirmation
            elif high[i] >= camarilla_H1_12h[i] and close[i] < camarilla_H1_12h[i] and adx_14_1d_12h[i] > 20 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches H4 (strong resistance) or ADX < 15 (trend weakening)
            if high[i] >= camarilla_H4_12h[i] or adx_14_1d_12h[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches L4 (strong support) or ADX < 15 (trend weakening)
            if low[i] <= camarilla_L4_12h[i] or adx_14_1d_12h[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_ADX14_Volume"
timeframe = "12h"
leverage = 1.0