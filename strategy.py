#!/usr/bin/env python3
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
    
    # Get weekly data for market context
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for long-term trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using previous week's data)
    pivot_1w = (high_1d + low_1d + close_1d) / 3.0  # Simplified: using daily OHLC for weekly pivot
    r1_1w = 2 * pivot_1w - low_1d
    s1_1w = 2 * pivot_1w - high_1d
    r2_1w = pivot_1w + (high_1d - low_1d)
    s2_1w = pivot_1w - (high_1d - low_1d)
    r3_1w = high_1d + 2 * (pivot_1w - low_1d)
    s3_1w = low_1d - 2 * (high_1d - pivot_1w)
    r4_1w = pivot_1w + 3 * (high_1d - low_1d)
    s4_1w = pivot_1w - 3 * (high_1d - low_1d)
    
    # Shift to use previous week's pivots (avoid look-ahead)
    r4_1w_prev = np.roll(r4_1w, 1)
    s4_1w_prev = np.roll(s4_1w, 1)
    r3_1w_prev = np.roll(r3_1w, 1)
    s3_1w_prev = np.roll(s3_1w, 1)
    r4_1w_prev[0] = np.nan
    s4_1w_prev[0] = np.nan
    r3_1w_prev[0] = np.nan
    s3_1w_prev[0] = np.nan
    
    # Align weekly pivot levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1w, r4_1w_prev)
    s4_6h = align_htf_to_ltf(prices, df_1w, s4_1w_prev)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3_1w_prev)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3_1w_prev)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (6h * 24 = 6d)
    volume_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need volume MA24, ATR MA20, and weekly pivots
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma24[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma20[i]) or 
            np.isnan(r4_6h[i]) or 
            np.isnan(s4_6h[i]) or
            np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 24-period average
        volume_filter = volume[i] > (2.0 * volume_ma24[i])
        # Volatility filter: ATR > ATR MA20 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma20[i]
        # Trend filter: price above/below weekly EMA200
        uptrend = close[i] > ema200_1w_aligned[i]
        downtrend = close[i] < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R4 with volume, volatility, and uptrend
            if close[i] > r4_6h[i] and volume_filter and volatility_filter and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume, volatility, and downtrend
            elif close[i] < s4_6h[i] and volume_filter and volatility_filter and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R3 or volatility drops or trend changes
            if close[i] < r3_6h[i] or not volatility_filter or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S3 or volatility drops or trend changes
            if close[i] > s3_6h[i] or not volatility_filter or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R4_S4_Breakout_Vol_Trend"
timeframe = "6h"
leverage = 1.0