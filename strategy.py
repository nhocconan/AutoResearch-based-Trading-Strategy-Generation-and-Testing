#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and choppiness regime.
Long when price breaks above R1 with 1d EMA34 uptrend and chop < 61.8 (trending).
Short when price breaks below S1 with 1d EMA34 downtrend and chop < 61.8.
Exit on opposite band touch or trend reversal.
Uses discrete sizing (0.25) to minimize fees. Target: 30-60 trades/year.
Works in bull via trend-following breakouts, in bear via mean reversion at bands when chop high.
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
    volume = prices['volume'].values
    
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
    R4_4h = np.full(len(close_4h), np.nan)
    S4_4h = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        # Camarilla levels based on previous 4h bar's range
        high_prev = high_4h[i-1]
        low_prev = low_4h[i-1]
        close_prev = close_4h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R1_4h[i] = close_prev + (range_prev * 1.1 / 12)
            S1_4h[i] = close_prev - (range_prev * 1.1 / 12)
            R4_4h[i] = close_prev + (range_prev * 1.1 / 2)
            S4_4h[i] = close_prev - (range_prev * 1.1 / 2)
    
    # Align Camarilla levels to original timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    R4_4h_aligned = align_htf_to_ltf(prices, df_4h, R4_4h)
    S4_4h_aligned = align_htf_to_ltf(prices, df_4h, S4_4h)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Choppiness regime: chop < 61.8 = trending (favor breakouts), chop > 61.8 = ranging
    # Calculate chop on 1h timeframe for better regime detection
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1 = np.abs(high_1h[1:] - low_1h[1:])
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1h = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    ll_1h = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr14) / (hh - ll)) / log10(14)
    chop_1h = np.full(len(close_1h), np.nan)
    for i in range(14, len(close_1h)):
        if atr_1h[i] > 0 and hh_1h[i] > ll_1h[i]:
            sum_atr = np.nansum(atr_1h[i-13:i+1])
            chop_1h[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10((hh_1h[i] - ll_1h[i]) + 1e-10)
        else:
            chop_1h[i] = 50.0  # neutral when invalid
    
    # Align chop to original timeframe
    chop_1h_aligned = align_htf_to_ltf(prices, df_1h, chop_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_1h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1 with uptrend and trending regime (chop < 61.8)
            long_signal = (close[i] > R1_4h_aligned[i]) and (close[i] > ema_34_1d_aligned[i]) and (chop_1h_aligned[i] < 61.8)
            # Short: price breaks below S1 with downtrend and trending regime (chop < 61.8)
            short_signal = (close[i] < S1_4h_aligned[i]) and (close[i] < ema_34_1d_aligned[i]) and (chop_1h_aligned[i] < 61.8)
            
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
            # Exit conditions: price touches S1 or trend reverses or chop too high (range)
            exit_signal = (close[i] < S1_4h_aligned[i]) or (close[i] < ema_34_1d_aligned[i]) or (chop_1h_aligned[i] > 61.8)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price touches R1 or trend reverses or chop too high (range)
            exit_signal = (close[i] > R1_4h_aligned[i]) or (close[i] > ema_34_1d_aligned[i]) or (chop_1h_aligned[i] > 61.8)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter"
timeframe = "4h"
leverage = 1.0