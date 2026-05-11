#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_Pivot_Breakout_Trend
Hypothesis: Breakouts at weekly Camarilla pivot levels (H4/L4) with 1-week EMA trend filter and volume confirmation.
This strategy trades daily breakouts in the direction of the weekly trend, using weekly pivot levels for precise entry.
Works in both bull and bear markets by aligning with higher timeframe trend. Volume filters out false breakouts.
Low trade frequency (~10-25 trades/year) reduces fee drag significantly.
"""

name = "1d_Weekly_Camarilla_Pivot_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

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
    
    # === Weekly Data for Trend and Pivot Levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_series = pd.Series(close_1w)
    
    # Weekly EMA50 for trend filter
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly Camarilla pivot levels: H4, L4
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    camarilla_h4 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_l4 = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align pivot levels to daily timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # === Volume Filter (1.5x 20-period EMA on daily) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly EMA and pivot calculation)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above weekly H4 with uptrend and volume
            if (close[i] > camarilla_h4_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below weekly L4 with downtrend and volume
            elif (close[i] < camarilla_l4_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below weekly L4 (mean reversion to pivot)
            if close[i] < camarilla_l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above weekly H4 (mean reversion to pivot)
            if close[i] > camarilla_h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals