#!/usr/bin/env python3
"""
12H_KAMA_Trend_RSI_ChopFilter
Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) for trend direction on 12h,
filtered by RSI extremes and Choppiness Index regime filter. Designed for 12h timeframe
to capture trend continuation with low trade frequency (target: 12-37 trades/year).
Works in both bull and bear markets by following adaptive trend direction and avoiding
choppy markets. Uses discrete position sizing (0.25) to minimize fee churn.
"""

name = "12H_KAMA_Trend_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Choppiness Index (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period) on weekly
    atr_1w = np.zeros(len(df_1w))
    tr_1w = np.maximum(np.maximum(df_1w['high'] - df_1w['low'],
                                  np.abs(df_1w['high'] - df_1w['close'].shift(1))),
                         np.abs(df_1w['low'] - df_1w['close'].shift(1)))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    max_high_1w = pd.Series(df_1w['high']).rolling(window=14, min_periods=14).max().values
    min_low_1w = pd.Series(df_1w['low']).rolling(window=14, min_periods=14).min().values
    
    chop_raw = 100 * np.log10(np.sum(atr_1w) / (max_high_1w - min_low_1w)) / np.log10(14)
    chop_1w = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on daily
    delta = df_1d['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 12h data for KAMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate KAMA(10,2,30) on 12h
    close_12h = df_12h['close']
    er = np.abs(close_12h.diff(10)) / (
        close_12h.diff(1).abs().rolling(window=10, min_periods=1).sum()
    )
    er = er.fillna(0).values
    sc = (er * (2/2 - 1/30) + 1/30) ** 2
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h.iloc[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h.iloc[i] - kama[i-1])
    kama_12h = kama
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(chop_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(kama_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_1w_aligned[i] < 38.2
        
        # Trend direction: price above/below KAMA
        price_above_kama = close[i] > kama_12h_aligned[i]
        price_below_kama = close[i] < kama_12h_aligned[i]
        
        if position == 0:
            # Long entry: price above KAMA + RSI not overbought + trending market
            if (price_above_kama and 
                rsi_1d_aligned[i] < 70 and 
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA + RSI not oversold + trending market
            elif (price_below_kama and 
                  rsi_1d_aligned[i] > 30 and 
                  is_trending):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or RSI overbought or choppy market
            if (price_below_kama or 
                rsi_1d_aligned[i] > 70 or 
                not is_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or RSI oversold or choppy market
            if (price_above_kama or 
                rsi_1d_aligned[i] < 30 or 
                not is_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals