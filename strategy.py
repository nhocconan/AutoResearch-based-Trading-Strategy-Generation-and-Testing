#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Trend_v1
Hypothesis: On daily timeframe, use KAMA for trend direction, RSI for momentum confirmation,
and weekly trend filter (price vs weekly SMA20) to avoid counter-trend trades.
Enters long when KAMA rising, RSI > 50, and price above weekly SMA20.
Enters short when KAMA falling, RSI < 50, and price below weekly SMA20.
Exits when trend reverses. Designed for low trade frequency (<20 trades/year) by requiring
alignment across multiple timeframes. Works in bull via long trends, in bear via short trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    # ER (Efficiency Ratio) = |change| / volatility
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    diff = np.diff(close_1d, prepend=close_1d[0])
    abs_diff = np.abs(diff)
    change_over_period = np.abs(np.diff(close_1d, 10))  # 10-period change
    sum_abs_diff = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        sum_abs_diff[i] = sum_abs_diff[i-1] + abs_diff[i]
        if i >= 10:
            sum_abs_diff[i] -= abs_diff[i-10]
    er = np.where(sum_abs_diff > 0, change_over_period / sum_abs_diff, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI (14-period) - momentum confirmation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === WEEKLY DATA (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly SMA20 for trend filter
    close_s = pd.Series(close_1w)
    sma20_1w = close_s.rolling(window=20, min_periods=20).mean().values
    
    # Align weekly SMA20 to daily
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # warmup period
        # Skip if not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(sma20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filters
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        price_above_weekly_sma = close[i] > sma20_1w_aligned[i]
        price_below_weekly_sma = close[i] < sma20_1w_aligned[i]
        
        # Momentum confirmation
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Entry conditions
        long_entry = kama_rising and rsi_bullish and price_above_weekly_sma
        short_entry = kama_falling and rsi_bearish and price_below_weekly_sma
        
        # Exit conditions (trend reversal)
        exit_long = not (kama_rising and rsi_bullish and price_above_weekly_sma)
        exit_short = not (kama_falling and rsi_bearish and price_below_weekly_sma)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals