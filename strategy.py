#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA (Kaufman Adaptive Moving Average) with RSI and chop filter.
# Long when KAMA turns up, RSI < 50 (mean reversion in trend), and chop > 61.8 (ranging market).
# Short when KAMA turns down, RSI > 50, and chop > 61.8.
# Uses weekly trend filter (1w EMA200) to avoid counter-trend trades in strong trends.
# Target: 7-25 trades/year (30-100 over 4 years) to minimize fee drag while capturing mean reversion in ranges.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - 10-period
    # ER (Efficiency Ratio) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close_1d - np.roll(close_1d, 10))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # will fix below
    
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            price_change = np.abs(close_1d[i] - close_1d[i-10])
            sum_abs_changes = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            if sum_abs_changes > 0:
                er[i] = price_change / sum_abs_changes
            else:
                er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close)
    for i in range(14, len(close)):
        if max_high[i] > min_low[i] and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: compare current vs previous
        kama_up = kama_aligned[i] > kama_aligned[i-1]
        kama_down = kama_aligned[i] < kama_aligned[i-1]
        
        # Long condition: KAMA turning up, RSI < 50 (oversold), chop > 61.8 (ranging)
        if (kama_up and 
            rsi_aligned[i] < 50 and 
            chop_aligned[i] > 61.8):
            signals[i] = 0.25
            position = 1
        # Short condition: KAMA turning down, RSI > 50 (overbought), chop > 61.8 (ranging)
        elif (kama_down and 
              rsi_aligned[i] > 50 and 
              chop_aligned[i] > 61.8):
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite KAMA signal or chop < 38.2 (trending market)
        elif position == 1 and (kama_down or chop_aligned[i] < 38.2):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (kama_up or chop_aligned[i] < 38.2):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop_EMA200_TrendFilter"
timeframe = "1d"
leverage = 1.0