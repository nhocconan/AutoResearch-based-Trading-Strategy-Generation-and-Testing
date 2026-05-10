#!/usr/bin/env python3
# 12H_Camarilla_Pivot_S1R1_Breakout_1dTrend_Filter
# Hypothesis: Camarilla pivot levels on daily chart provide key support/resistance levels. Price breaking above S1 or below R1 with daily trend filter captures institutional order flow. Designed for low trade frequency (~20-40/year) with discrete sizing (0.25) to minimize fee drag and work in both bull and bear markets.

name = "12H_Camarilla_Pivot_S1R1_Breakout_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels (S1, R1)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: S1 = C - (H-L)*1.12/12, R1 = C + (H-L)*1.12/12
    camarilla_S1 = close_1d - (high_1d - low_1d) * 1.12 / 12
    camarilla_R1 = close_1d + (high_1d - low_1d) * 1.12 / 12
    
    # Align Camarilla levels to 12h timeframe (no additional delay needed for pivot levels)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    
    # Daily trend filter: EMA 50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_R1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get daily close for trend determination
        close_1d_series = pd.Series(close_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_series.values)
        
        is_uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        is_downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above S1 and daily uptrend
            if close[i] > camarilla_S1_aligned[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below R1 and daily downtrend
            elif close[i] < camarilla_R1_aligned[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S1 or daily trend turns down
            if close[i] < camarilla_S1_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above R1 or daily trend turns up
            if close[i] > camarilla_R1_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals