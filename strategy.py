#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_With_RSI_and_Chop_Filter_v1
Hypothesis: KAMA identifies trend direction on daily chart, RSI filters for overbought/oversold within trend, and Chop filter ensures we only trade in trending markets (Chop < 38.2). Works in bull by catching trend continuations, in bear by avoiding false signals during consolidation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for KAMA and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d)).cumsum()
    volatility = np.diff(np.concatenate([[0], volatility]))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 1/30) + 1/30) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Chop calculation (14-period)
    atr = np.full_like(close_1d, np.nan)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    range_hl = max_high - min_low
    chop = np.where(range_hl != 0, 100 * np.log10(sum_atr / range_hl) / np.log10(14), 50)
    
    # RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align all indicators to 1d timeframe (same as prices)
    kama_aligned = kama  # already 1d
    rsi_aligned = rsi    # already 1d
    chop_aligned = chop  # already 1d
    
    # Load weekly data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA34 on weekly
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Trend condition: price > KAMA for long, price < KAMA for short
        price_above_kama = price > kama_aligned[i]
        price_below_kama = price < kama_aligned[i]
        
        # Chop filter: only trade when market is trending (Chop < 38.2)
        chop_trending = chop_aligned[i] < 38.2
        
        # RSI filter: avoid extremes
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        # Weekly trend confirmation
        if i >= 31:
            ema34_prev = ema34_1w_aligned[i-1]
            ema34_curr = ema34_1w_aligned[i]
            weekly_uptrend = ema34_curr > ema34_prev
            weekly_downtrend = ema34_curr < ema34_prev
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long: price > KAMA + Chop trending + RSI not overbought + weekly uptrend
            if price_above_kama and chop_trending and rsi_not_overbought and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA + Chop trending + RSI not oversold + weekly downtrend
            elif price_below_kama and chop_trending and rsi_not_oversold and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA OR Chop becomes choppy
            if price < kama_aligned[i] or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA OR Chop becomes choppy
            if price > kama_aligned[i] or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_KAMA_Trend_With_RSI_and_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0