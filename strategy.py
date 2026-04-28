#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_1wTrend_Filter
Hypothesis: Daily KAMA direction filter with weekly trend confirmation. Uses KAMA's adaptive smoothing to reduce whipsaw in sideways markets and catch strong trends. Weekly trend filter ensures alignment with higher timeframe momentum, reducing false signals. Works in bull markets (long when both timeframes bullish) and bear markets (short when both bearish). Target: 15-30 trades/year to minimize fee drag while capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close(t) - close(t-10)|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum |close(t) - close(t-1)| over 10 periods
    # Fix dimensions: change starts at index 10, volatility from index 1 to 10
    change_full = np.concatenate([np.full(10, np.nan), change])
    volatility_full = np.concatenate([np.full(1, np.nan), volatility])
    # Pad volatility to match length
    if len(volatility_full) < len(close_1d):
        volatility_full = np.concatenate([volatility_full, np.full(len(close_1d) - len(volatility_full), np.nan)])
    elif len(volatility_full) > len(close_1d):
        volatility_full = volatility_full[:len(close_1d)]
    
    # Calculate ER, handling division by zero
    er = np.where(volatility_full != 0, change_full / volatility_full, 0)
    # Smoothing constants: fastest SC=2/(2+1)=0.667, slowest SC=2/(30+1)=0.0645
    sc = (er * (0.667 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to daily timeframe (already daily, but use for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: price above/below KAMA
        kama_bullish = close[i] > kama_aligned[i]
        kama_bearish = close[i] < kama_aligned[i]
        
        # Weekly trend filter: price above/below weekly EMA34
        weekly_bullish = close[i] > ema_34_1w_aligned[i]
        weekly_bearish = close[i] < ema_34_1w_aligned[i]
        
        # Entry conditions: both timeframes aligned
        long_entry = kama_bullish and weekly_bullish
        short_entry = kama_bearish and weekly_bearish
        
        # Exit conditions: divergence between timeframes
        long_exit = not kama_bullish or not weekly_bullish
        short_exit = not kama_bearish or not weekly_bullish
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Trend_With_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0