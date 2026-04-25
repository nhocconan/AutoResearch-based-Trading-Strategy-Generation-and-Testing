#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_WeeklyTrend
Hypothesis: Daily KAMA trend direction + RSI momentum + weekly trend filter + choppiness regime.
Long when KAMA rising, RSI > 50, weekly trend up, and market not too choppy (CHOP < 61.8).
Short when KAMA falling, RSI < 50, weekly trend down, and market not too choppy.
Exit on opposite KAMA direction or trend reversal.
Uses discrete sizing (0.25) to minimize fees. Target: 15-30 trades/year.
Works in bull via trend following, in bear via avoiding false signals in choppy markets.
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
    
    # Get 1d data for KAMA and RSI calculations (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation: |net change| / sum(|abs changes|) over period
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):  # min_periods=10 for ER
        net_change = abs(close_1d[i] - close_1d[i-10])
        sum_abs_changes = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
        if sum_abs_changes > 0:
            er[i] = net_change / sum_abs_changes
        else:
            er[i] = 0
    
    # SC = [ER * (fastest_SC - slowest_SC) + slowest_SC]^2
    fastest_sc = 2 / (2 + 1)   # EMA(2)
    slowest_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on daily (CHOP(14))
    # True Range
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3[0] = np.abs(high_1d[0] - close_1d[0])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if tr_sum[i] > 0 and hh[i] != ll[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align KAMA, RSI, CHOP to original timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 1w data for trend filter (EMA34 on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, weekly uptrend, not too choppy
            kama_rising = kama_aligned[i] > kama_aligned[i-1]
            rsi_bullish = rsi_aligned[i] > 50
            weekly_uptrend = close[i] > ema_34_1w_aligned[i]
            not_choppy = chop_aligned[i] < 61.8
            
            if kama_rising and rsi_bullish and weekly_uptrend and not_choppy:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI < 50, weekly downtrend, not too choppy
            elif (not kama_rising) and (rsi_aligned[i] < 50) and (close[i] < ema_34_1w_aligned[i]) and (chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: KAMA falling or weekly trend turns down
            exit_signal = (not (kama_aligned[i] > kama_aligned[i-1])) or (close[i] < ema_34_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: KAMA rising or weekly trend turns up
            exit_signal = (kama_aligned[i] > kama_aligned[i-1]) or (close[i] > ema_34_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_WeeklyTrend"
timeframe = "1d"
leverage = 1.0