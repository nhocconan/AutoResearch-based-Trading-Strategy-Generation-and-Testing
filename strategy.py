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
    
    # Get weekly data for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 21-period EMA on weekly close for trend direction
    close_1w_series = pd.Series(close_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Get daily data for pivot points (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Use previous day's pivots (avoid look-ahead)
    r1_1d_prev = np.roll(r1_1d, 1)
    s1_1d_prev = np.roll(s1_1d, 1)
    r2_1d_prev = np.roll(r2_1d, 1)
    s2_1d_prev = np.roll(s2_1d, 1)
    r3_1d_prev = np.roll(r3_1d, 1)
    s3_1d_prev = np.roll(s3_1d, 1)
    r1_1d_prev[0] = np.nan
    s1_1d_prev[0] = np.nan
    r2_1d_prev[0] = np.nan
    s2_1d_prev[0] = np.nan
    r3_1d_prev[0] = np.nan
    s3_1d_prev[0] = np.nan
    
    # Align daily pivot levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d_prev)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d_prev)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_1d_prev)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_1d_prev)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d_prev)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d_prev)
    
    # Volume confirmation: current volume > 1.3 * 20-period average (reduced threshold)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need volume MA20, ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r1_12h[i]) or 
            np.isnan(s1_12h[i]) or
            np.isnan(r2_12h[i]) or
            np.isnan(s2_12h[i]) or
            np.isnan(r3_12h[i]) or
            np.isnan(s3_12h[i]) or
            np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        # Weekly trend filter: price above/below weekly EMA21
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume, volatility and uptrend
            if (close[i] > r2_12h[i] and volume_filter and volatility_filter and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume, volatility and downtrend
            elif (close[i] < s2_12h[i] and volume_filter and volatility_filter and downtrend):
                signals[i] = -0.25
                position = -1
            # Long mean reversion: price touches S3 with volume and volatility in uptrend
            elif (close[i] < s3_12h[i] and volume_filter and volatility_filter and uptrend):
                signals[i] = 0.20
                position = 1
            # Short mean reversion: price touches R3 with volume and volatility in downtrend
            elif (close[i] > r3_12h[i] and volume_filter and volatility_filter and downtrend):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or volatility drops or trend changes
            if close[i] < r1_12h[i] or not volatility_filter or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 or volatility drops or trend changes
            if close[i] > s1_12h[i] or not volatility_filter or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R2_S2_WeeklyTrendFilter_Volume"
timeframe = "12h"
leverage = 1.0