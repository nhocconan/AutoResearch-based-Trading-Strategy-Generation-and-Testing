#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolatilityFilter
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volatility regime filter.
Long when price breaks above R1 with 4h EMA50 uptrend and 1d ATR ratio > 0.8 (normal volatility).
Short when price breaks below S1 with 4h EMA50 downtrend and 1d ATR ratio > 0.8.
Exit on opposite band touch or trend reversal.
Uses discrete sizing (0.20) to minimize fee churn. Target: 15-30 trades/year.
Works in bull via trend-following breakouts, in bear via reduced false breakouts in low volatility.
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
    
    # Get 4h data for Camarilla calculations and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar (based on previous bar)
    R1_4h = np.full(len(close_4h), np.nan)
    S1_4h = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        # Camarilla levels based on previous 4h bar's range
        high_prev = high_4h[i-1]
        low_prev = low_4h[i-1]
        close_prev = close_4h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R1_4h[i] = close_prev + (range_prev * 1.1 / 12)
            S1_4h[i] = close_prev - (range_prev * 1.1 / 12)
    
    # Align Camarilla levels to original timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    
    # Get 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volatility filter (ATR ratio)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_1d = np.concatenate([[np.nan], pd.Series(tr).rolling(window=14, min_periods=14).mean().values])
    
    # Calculate 1d ATR ratio (current ATR / 50-period MA of ATR)
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volatility filter: only trade in normal/high volatility (ATR ratio > 0.8)
        vol_filter = atr_ratio_aligned[i] > 0.8
        
        if position == 0:
            # Long: price breaks above R1 with uptrend and volume filter
            long_signal = (close[i] > R1_4h_aligned[i]) and (close[i] > ema_50_4h_aligned[i]) and vol_filter
            # Short: price breaks below S1 with downtrend and volume filter
            short_signal = (close[i] < S1_4h_aligned[i]) and (close[i] < ema_50_4h_aligned[i]) and vol_filter
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions: price touches S1 or trend reverses
            exit_signal = (close[i] < S1_4h_aligned[i]) or (close[i] < ema_50_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions: price touches R1 or trend reverses
            exit_signal = (close[i] > R1_4h_aligned[i]) or (close[i] > ema_50_4h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolatilityFilter"
timeframe = "1h"
leverage = 1.0