#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: Daily KAMA direction with RSI extremes and choppiness regime filter.
Enters long when KAMA turns up, RSI < 30, and choppy market (CHOP > 61.8).
Enters short when KAMA turns down, RSI > 70, and choppy market (CHOP > 61.8).
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 30-100 total trades over 4 years.
Works in ranging markets where mean reversion occurs, avoids strong trends via chop filter.
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
    
    # Load weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 10-period volatility
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    fastest = 2/(2+1)
    slowest = 2/(30+1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i-1] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Daily RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Daily Choppiness Index(14)
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(np.diff(close_1d))
    tr2 = np.abs(np.subtract(close_1d[1:], high[:-1]))
    tr3 = np.abs(np.subtract(close_1d[1:], low[:-1]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr[1:] = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr[0] = atr[1] if len(atr) > 1 else 0
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    chop = np.divide(np.log10(sum_atr14) * 100, np.log10(max_high - min_low), out=np.full_like(close_1d, 50.0), where=(max_high - min_low)!=0)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 10-period ER + 14-period RSI/CHOP + weekly EMA)
    start_idx = 20  # conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: KAMA turning up + RSI oversold + choppy market
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        rsi_oversold = rsi_aligned[i] < 30
        choppy = chop_aligned[i] > 61.8
        # Avoid strong uptrend (weekly EMA filter)
        not_strong_uptrend = close[i] < ema_50_1w_aligned[i]
        
        if kama_rising and rsi_oversold and choppy and not_strong_uptrend:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: KAMA turning down + RSI overbought + choppy market
        elif kama_aligned[i] < kama_aligned[i-1] and rsi_aligned[i] > 70 and chop_aligned[i] > 61.8 and close[i] > ema_50_1w_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite KAMA direction or chop low (trending market)
        elif position == 1 and (kama_aligned[i] < kama_aligned[i-1] or chop_aligned[i] < 38.2):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (kama_aligned[i] > kama_aligned[i-1] or chop_aligned[i] < 38.2):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0