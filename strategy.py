#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Filter
Hypothesis: Camarilla R1/S1 breakout on 12h timeframe with 1d EMA50 trend filter.
Only trade breakouts in direction of daily trend to avoid counter-trend whipsaws.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
Works in both bull and bear markets via trend alignment - avoids trading against the higher timeframe trend.
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
    
    # Get 1d data for HTF trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels (R1, S1) from previous 1d candle
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align HTF indicators to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for Camarilla breakout signals with trend filter
            # Long: price breaks above R1 in uptrend (close > EMA50)
            # Short: price breaks below S1 in downtrend (close < EMA50)
            long_signal = (close[i] > r1_aligned[i]) and (close[i] > ema50_aligned[i])
            short_signal = (close[i] < s1_aligned[i]) and (close[i] < ema50_aligned[i])
            
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
            # Exit when price moves back below EMA50 (trend reversal)
            exit_signal = close[i] < ema50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above EMA50 (trend reversal)
            exit_signal = close[i] > ema50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0