#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_v1
Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter. Camarilla levels provide mean-reversion structure while weekly EMA34 ensures trading with higher timeframe trend to avoid counter-trend whipsaws. Designed for low frequency (target 7-25 trades/year) to minimize fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data ONCE before loop for Camarilla levels (structure)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior daily bar (completed bar only)
    # Camarilla R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    range_1d = df_1d['high'] - df_1d['low']
    camarilla_r1 = df_1d['close'] + 1.1 * range_1d / 12
    camarilla_s1 = df_1d['close'] - 1.1 * range_1d / 12
    
    # Align Camarilla levels to daily timeframe (no additional delay needed for structure)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Load weekly data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter (needs completed weekly candle)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1w = np.where(ema_34_1w_aligned > 0, 
                        np.where(close > ema_34_1w_aligned, 1, -1), 
                        0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(trend_1w[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with weekly trend filter
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND weekly uptrend
            if close[i] > camarilla_r1_aligned[i] and trend_1w[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 AND weekly downtrend
            elif close[i] < camarilla_s1_aligned[i] and trend_1w[i] == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S1 OR weekly trend turns down
            if close[i] < camarilla_s1_aligned[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R1 OR weekly trend turns up
            if close[i] > camarilla_r1_aligned[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_v1"
timeframe = "1d"
leverage = 1.0