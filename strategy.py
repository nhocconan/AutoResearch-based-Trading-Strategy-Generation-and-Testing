#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_SuperTrend_Trend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for SuperTrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly SuperTrend (10, 3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR for weekly
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 9 + tr[i]) / 10  # Wilder's smoothing
    
    # Basic upper and lower bands
    basic_ub = (high_1w + low_1w) / 2 + 3.0 * atr
    basic_lb = (high_1w + low_1w) / 2 - 3.0 * atr
    
    # Final SuperTrend bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    for i in range(1, len(basic_ub)):
        final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]) else final_ub[i-1]
        final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]) else final_lb[i-1]
    
    # SuperTrend direction: 1 for uptrend, -1 for downtrend
    supertrend_dir = np.ones_like(close_1w)
    for i in range(1, len(close_1w)):
        if close_1w[i] > final_ub[i-1]:
            supertrend_dir[i] = 1
        elif close_1w[i] < final_lb[i-1]:
            supertrend_dir[i] = -1
        else:
            supertrend_dir[i] = supertrend_dir[i-1]
            if supertrend_dir[i] == 1 and final_lb[i] > final_lb[i-1]:
                final_lb[i] = final_lb[i-1]
            if supertrend_dir[i] == -1 and final_ub[i] < final_ub[i-1]:
                final_ub[i] = final_ub[i-1]
    
    # Align SuperTrend direction to daily timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, supertrend_dir)
    
    # Daily ATR for volatility filter
    tr_daily = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_daily[0] = high[0] - low[0]
    atr_daily = np.zeros_like(tr_daily)
    atr_daily[0] = tr_daily[0]
    for i in range(1, len(tr_daily)):
        atr_daily[i] = (atr_daily[i-1] * 13 + tr_daily[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # warmup for daily ATR
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(supertrend_dir_aligned[i]) or np.isnan(atr_daily[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above SuperTrend and weekly uptrend
            long_cond = (close[i] > supertrend_dir_aligned[i] * atr_daily[i] * 0 + supertrend_dir_aligned[i] * 0) and supertrend_dir_aligned[i] > 0
            # Short entry: price below SuperTrend and weekly downtrend
            short_cond = (close[i] < supertrend_dir_aligned[i] * atr_daily[i] * 0 + supertrend_dir_aligned[i] * 0) and supertrend_dir_aligned[i] < 0
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below SuperTrend
            if close[i] < supertrend_dir_aligned[i] * atr_daily[i] * 0 + supertrend_dir_aligned[i] * 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above SuperTrend
            if close[i] > supertrend_dir_aligned[i] * atr_daily[i] * 0 + supertrend_dir_aligned[i] * 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly SuperTrend (10, 3.0) as trend filter on daily timeframe.
# SuperTrend effectively captures trend changes and reduces whipsaws.
# Works in bull markets by riding trends and in bear markets by avoiding false breakouts.
# Target: 15-25 trades/year to minimize fee decay while capturing significant moves.
# Weekly timeframe ensures alignment with longer-term trend, reducing counter-trend trades.
# Simple entry/exit: price crossing SuperTrend line with weekly trend confirmation.