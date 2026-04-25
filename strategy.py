#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and choppiness regime filter.
Long when price breaks above R1 with 1d EMA50 uptrend and CHOP > 61.8 (range regime).
Short when price breaks below S1 with 1d EMA50 downtrend and CHOP > 61.8.
Exit on opposite band touch or trend reversal.
Uses discrete sizing (0.25) to minimize fee churn. Target: 30-60 trades/year.
Works in bull via trend-following breakouts, in bear via mean reversion at extreme bands in range regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for Camarilla calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar (based on previous bar)
    R1_4h = np.full(len(close_4h), np.nan)
    S1_4h = np.full(len(close_4h), np.nan)
    R3_4h = np.full(len(close_4h), np.nan)
    S3_4h = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        # Camarilla levels based on previous 4h bar's range
        high_prev = high_4h[i-1]
        low_prev = low_4h[i-1]
        close_prev = close_4h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R1_4h[i] = close_prev + (range_prev * 1.1 / 12)
            S1_4h[i] = close_prev - (range_prev * 1.1 / 12)
            R3_4h[i] = close_prev + (range_prev * 1.1 / 4)
            S3_4h[i] = close_prev - (range_prev * 1.1 / 4)
    
    # Align Camarilla levels to original timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    R3_4h_aligned = align_htf_to_ltf(prices, df_4h, R3_4h)
    S3_4h_aligned = align_htf_to_ltf(prices, df_4h, S3_4h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d data for choppiness regime filter (CHOP(14))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # ATR = TR / 14 (approximation for CHOP)
    atr_14 = tr_14 / 14
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop_raw = np.where(range_14 > 0, tr_14 / range_14, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade in range regimes (CHOP > 61.8)
        in_range_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R1 with uptrend and in range regime
            long_signal = (close[i] > R1_4h_aligned[i]) and (close[i] > ema_50_1d_aligned[i]) and in_range_regime
            # Short: price breaks below S1 with downtrend and in range regime
            short_signal = (close[i] < S1_4h_aligned[i]) and (close[i] < ema_50_1d_aligned[i]) and in_range_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: price touches S1 or trend reverses
            exit_signal = (close[i] < S1_4h_aligned[i]) or (close[i] < ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price touches R1 or trend reverses
            exit_signal = (close[i] > R1_4h_aligned[i]) or (close[i] > ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter"
timeframe = "4h"
leverage = 1.0