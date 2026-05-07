#!/usr/bin/env python3
name = "1d_1w_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly KAMA direction (14-period)
    # Efficiency Ratio
    change = np.abs(np.diff(df_1w['close'], prepend=df_1w['close'][0]))
    volatility = np.abs(np.diff(df_1w['close'])).cumsum()
    volatility = np.diff(volatility, prepend=0)
    er = change / (volatility + 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(df_1w['close'])
    kama[0] = df_1w['close'][0]
    for i in range(1, len(df_1w)):
        kama[i] = kama[i-1] + sc[i] * (df_1w['close'][i] - kama[i-1])
    kama_dir = kama > np.roll(kama, 1)
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Choppiness Index(14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum(axis=0) / (highest_high - lowest_low)) / np.log10(14) if False else \
           100 * np.log10(pd.Series(tr).rolling(14).sum() / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # Align weekly KAMA direction to daily
    kama_dir_aligned = align_htf_to_ltf(prices, df_1w, kama_dir.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Wait for RSI and chop
    
    for i in range(start_idx, n):
        if np.isnan(kama_dir_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend, RSI > 50, chop < 61.8 (trending)
            if kama_dir_aligned[i] and rsi[i] > 50 and chop[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend, RSI < 50, chop < 61.8 (trending)
            elif not kama_dir_aligned[i] and rsi[i] < 50 and chop[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: weekly trend change or RSI < 40 or chop > 61.8 (range)
            if not kama_dir_aligned[i] or rsi[i] < 40 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: weekly trend change or RSI > 60 or chop > 61.8 (range)
            if kama_dir_aligned[i] or rsi[i] > 60 or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily KAMA direction from weekly trend with RSI and chop filter
# - Weekly KAMA trend determines primary direction (avoid counter-trend trades)
# - Daily RSI > 50 for long, < 50 for short ensures momentum alignment
# - Chop < 61.8 filters for trending markets, avoids whipsaws in ranging
# - Works in both bull (buy in weekly uptrend) and bear (sell in weekly downtrend)
# - Exit when trend changes, RSI reverses, or market becomes choppy
# - Position size 0.25 targets ~15-25 trades/year, avoiding fee drag
# - Uses actual weekly data via mtf_data to prevent look-ahead
# - Designed for 1d timeframe to capture multi-day trends with low turnover