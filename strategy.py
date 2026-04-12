#!/usr/bin/env python3
"""
6h_1d_Camarilla_Breakout_Momentum_v1
Hypothesis: Use daily Camarilla pivot levels with momentum confirmation on 6h.
Long when price breaks above H4 (daily) with RSI < 60, short when breaks below L4 (daily) with RSI > 40.
Focus on strong momentum breakouts to avoid false signals and reduce trade frequency.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drift.
Works in bull via breakouts, in bear via mean-reversion from extreme levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_Breakout_Momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Calculate daily Camarilla levels
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    daily_h4 = prev_close + 1.1 * range_val * 1.1 / 2
    daily_l4 = prev_close - 1.1 * range_val * 1.1 / 2
    
    # Align daily levels to 6h timeframe
    daily_h4_array = np.full(len(df_1d), daily_h4)
    daily_l4_array = np.full(len(df_1d), daily_l4)
    daily_h4_aligned = align_htf_to_ltf(prices, df_1d, daily_h4_array)
    daily_l4_aligned = align_htf_to_ltf(prices, df_1d, daily_l4_array)
    
    # RSI (14-period) for momentum filter
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(daily_h4_aligned[i]) or np.isnan(daily_l4_aligned[i]) or
            np.isnan(rsi_values[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with momentum filter
        long_breakout = close[i] > daily_h4_aligned[i] and rsi_values[i] < 60
        short_breakout = close[i] < daily_l4_aligned[i] and rsi_values[i] > 40
        
        # Exit conditions: return to pivot
        daily_pivot = (prev_high + prev_low + prev_close) / 3
        daily_pivot_array = np.full(len(df_1d), daily_pivot)
        daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot_array)
        
        long_exit = close[i] < daily_pivot_aligned[i]
        short_exit = close[i] > daily_pivot_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals