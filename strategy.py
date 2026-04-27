#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_MeanReversion_ChopFilter
Hypothesis: Daily KAMA trend direction combined with RSI mean reversion and choppy market filter.
Long when KAMA trending up AND RSI < 30 AND choppy market (CHOP > 61.8).
Short when KAMA trending down AND RSI > 70 AND choppy market (CHOP > 61.8).
Exit when RSI returns to neutral (40-60) or chop ends.
Designed for 1d timeframe to capture mean reversion in choppy markets while avoiding strong trends.
Works in bull markets (buy dips in uptrend chop) and bear markets (sell rallies in downtrend chop).
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
    
    # 1d KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 10-period sum of absolute changes
    # Fix array lengths: volatility needs to be same length as change
    volatility = pd.Series(np.abs(np.diff(close_1d))).rolling(window=10, min_periods=1).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.concatenate([[0.0], sc])  # align with close_1d
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily bars (already aligned since we computed on df_1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # KAMA trend: upward if today's KAMA > yesterday's KAMA
    kama_trend_up = np.zeros_like(kama_aligned, dtype=bool)
    kama_trend_up[1:] = kama_aligned[1:] > kama_aligned[:-1]
    kama_trend_down = np.zeros_like(kama_aligned, dtype=bool)
    kama_trend_down[1:] = kama_aligned[1:] < kama_aligned[:-1]
    
    # Daily RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[50.0], rsi])  # align with close_1d, first value neutral
    
    # Align RSI to daily bars
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Weekly chop filter (CHOP > 61.8 = choppy/ranging)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[0.0], tr])  # align with index 0
    
    # ATR(14)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    # Chopiness Index: CHOP = 100 * log10(sum(ATR14) / range14) / log10(14)
    chop = np.where(
        range_14 > 0,
        100 * np.log10(sum_atr / range_14) / np.log10(14),
        50.0  # neutral when range is zero
    )
    chop = np.concatenate([[50.0], chop])  # align with index 0
    
    # Align chop to weekly bars
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Choppy market condition: CHOP > 61.8
    chop_condition = chop_aligned > 61.8
    
    # RSI mean reversion levels
    rsi_oversold = 30
    rsi_overbought = 70
    rsi_neutral_low = 40
    rsi_neutral_high = 60
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for KAMA (~30d), RSI (~14d), CHOP (~14w)
    start_idx = max(30, 14, 14*7)  # approx bars needed
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        kama_prev = kama_aligned[i-1] if i > 0 else kama_val
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Flat - look for entry
            # Long: KAMA trending up AND RSI oversold AND choppy market
            # Short: KAMA trending down AND RSI overbought AND choppy market
            long_condition = (kama_val > kama_prev and 
                            rsi_val < rsi_oversold and 
                            chop_val > 61.8)
            short_condition = (kama_val < kama_prev and 
                             rsi_val > rsi_overbought and 
                             chop_val > 61.8)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long - exit when RSI returns to neutral OR chop ends
            if (rsi_val >= rsi_neutral_low and rsi_val <= rsi_neutral_high) or chop_val <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when RSI returns to neutral OR chop ends
            if (rsi_val >= rsi_neutral_low and rsi_val <= rsi_neutral_high) or chop_val <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_RSI_MeanReversion_ChopFilter"
timeframe = "1d"
leverage = 1.0