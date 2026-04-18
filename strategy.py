#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout + Volume Confirmation + ADX Filter
Hypothesis: Weekly pivot levels act as strong support/resistance. Breakouts above weekly R1 or below S1 with volume confirmation and ADX > 20 capture institutional moves. Works in both bull (breakouts up) and bear (breakouts down) markets. Low trade frequency due to strict weekly pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_wp(high, low, close):
    """Calculate weekly pivot points"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def calculate_adx(high, low, close, period=14):
    """Calculate ADX using Wilder smoothing"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder smoothing
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilder_smooth(tr, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / np.where(atr != 0, atr, 1)
    minus_di = 100 * wilder_smooth(minus_dm, period) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smooth(dx, period)
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pivot_w, r1_w, r2_w, r3_w, s1_w, s2_w, s3_w = calculate_wp(high_w, low_w, close_w)
    
    # Align weekly pivots to 6h
    pivot_w_a = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_a = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_a = align_htf_to_ltf(prices, df_w, s1_w)
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_a = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot_w_a[i]) or np.isnan(r1_w_a[i]) or np.isnan(s1_w_a[i]) or np.isnan(adx_1d_a[i]):
            signals[i] = 0.0
            continue
        
        pivot_val = pivot_w_a[i]
        r1_val = r1_w_a[i]
        s1_val = s1_w_a[i]
        adx_val = adx_1d_a[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: break above R1 with volume and ADX > 20
            if (close[i] > r1_val and 
                adx_val > 20 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume and ADX > 20
            elif (close[i] < s1_val and 
                  adx_val > 20 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot or ADX weakens
            if close[i] < pivot_val or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot or ADX weakens
            if close[i] > pivot_val or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_Volume_ADXFilter"
timeframe = "6h"
leverage = 1.0