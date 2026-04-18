#!/usr/bin/env python3
"""
4h Camarilla Pivot Reversal with Volume Spike and ADX Trend Filter
Hypothesis: Camarilla pivot levels (R1/S1) act as strong support/resistance in both bull and bear markets.
Price reversals at these levels with volume confirmation and ADX > 25 filter capture high-probability mean-reversion trades.
Works in trending and ranging markets by fading extremes at key pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index with proper smoothing"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_dm_smooth = np.zeros_like(high)
    minus_dm_smooth = np.zeros_like(high)
    
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    
    for i in range(1, len(high)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx = np.zeros_like(high)
    adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low) * 1.1/2
    # R3 = close + 1.5 * (high - low) * 1.1/4
    # R2 = close + 1.5 * (high - low) * 1.1/6
    # R1 = close + 1.5 * (high - low) * 1.1/12
    # S1 = close - 1.5 * (high - low) * 1.1/12
    # S2 = close - 1.5 * (high - low) * 1.1/6
    # S3 = close - 1.5 * (high - low) * 1.1/4
    # S4 = close - 1.5 * (high - low) * 1.1/2
    
    range_1d = high_1d - low_1d
    camarilla_multiplier = 1.5 * range_1d * 1.1
    
    r4 = close_1d + camarilla_multiplier * 0.5
    r3 = close_1d + camarilla_multiplier * 0.25
    r2 = close_1d + camarilla_multiplier * (1/6)
    r1 = close_1d + camarilla_multiplier * (1/12)
    s1 = close_1d - camarilla_multiplier * (1/12)
    s2 = close_1d - camarilla_multiplier * (1/6)
    s3 = close_1d - camarilla_multiplier * 0.25
    s4 = close_1d - camarilla_multiplier * 0.5
    
    # For reversal strategy, we focus on R1/S1 as primary levels
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate ADX on daily timeframe for trend filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 19:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX > 20 (allows both trending and ranging markets)
        # Lower threshold to capture more opportunities while avoiding choppy markets
        trending_or_ranging = adx_1d_aligned[i] > 20
        
        if position == 0:
            # Long reversal: price touches or goes below S1 with volume spike
            if (close[i] <= s1_aligned[i] and vol_spike[i] and trending_or_ranging):
                signals[i] = 0.25
                position = 1
            # Short reversal: price touches or goes above R1 with volume spike
            elif (close[i] >= r1_aligned[i] and vol_spike[i] and trending_or_ranging):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns above S1 (mean reversion complete) or volume drops
            if close[i] > s1_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns below R1 (mean reversion complete) or volume drops
            if close[i] < r1_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Reversal_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0