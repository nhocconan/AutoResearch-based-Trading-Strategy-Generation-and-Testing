#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1_S1_Breakout_Trend
Hypothesis: Use weekly trend filter with daily Camarilla R1/S1 breakouts for position entries.
In weekly uptrend (price above weekly EMA20), go long on break above daily R1.
In weekly downtrend (price below weekly EMA20), go short on break below daily S1.
Exit when price returns to daily Pivot or reverses weekly trend.
Weekly trend filter reduces whipsaw and improves trend alignment. Weekly timeframe provides
more stable trend signals than daily, reducing false breakouts in choppy markets.
Target: 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
Works in bull by buying breakouts in uptrend; works in bear by selling breakdowns in downtrend.
"""

name = "1d_1w_Camarilla_R1_S1_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    
    # --- Weekly Trend Filter: EMA20 ---
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # --- Daily Camarilla Levels (from previous day) ---
    high_1d_d = df_1d['high'].values
    low_1d_d = df_1d['low'].values
    close_1d_d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1, Pivot (using previous day's OHLC)
    camarilla_R1 = np.full(len(high_1d_d), np.nan)
    camarilla_S1 = np.full(len(high_1d_d), np.nan)
    camarilla_P = np.full(len(high_1d_d), np.nan)
    
    for i in range(1, len(high_1d_d)):
        # Previous day's OHLC
        ph = high_1d_d[i-1]
        pl = low_1d_d[i-1]
        pc = close_1d_d[i-1]
        # Camarilla formulas
        camarilla_P[i] = (ph + pl + pc) / 3.0
        camarilla_R1[i] = pc + (ph - pl) * 1.1 / 12
        camarilla_S1[i] = pc - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to daily timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_P_aligned = align_htf_to_ltf(prices, df_1d, camarilla_P)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20  # for weekly EMA20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(camarilla_P_aligned[i]) or np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close_1d[i] > ema20_1w_aligned[i]
        trend_down = close_1d[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend
            if trend_up and close_1d[i] > camarilla_R1_aligned[i]:
                # Long: weekly uptrend + break above daily R1
                signals[i] = 0.25
                position = 1
            elif trend_down and close_1d[i] < camarilla_S1_aligned[i]:
                # Short: weekly downtrend + break below daily S1
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: trend turns down OR price returns to daily Pivot
                if not trend_up or close_1d[i] < camarilla_P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend turns up OR price returns to daily Pivot
                if not trend_down or close_1d[i] > camarilla_P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals