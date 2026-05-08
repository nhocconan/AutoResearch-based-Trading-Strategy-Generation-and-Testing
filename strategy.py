#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA calculation (using Efficiency Ratio)
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate KAMA on 1d close
    kama_1d = kama(df_1d['close'].values, length=10, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # RSI calculation (14-period)
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = rsi(df_1d['close'].values, length=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Choppiness Index (14-period) on 1d
    def choppiness_index(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.max(tr1[:1]) if len(tr1) > 0 else 0], tr])
        atr = pd.Series(tr).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        
        highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        
        sum_atr = pd.Series(atr).rolling(window=length, min_periods=length).sum().values
        range_hl = highest_high - lowest_low
        cpi = 100 * np.log10(sum_atr / range_hl) / np.log10(length)
        return np.where(range_hl != 0, cpi, 50)
    
    chop_1d = choppiness_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, length=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above KAMA, RSI > 50, and chop < 61.8 (trending market)
            long_cond = (close[i] > kama_1d_aligned[i] and 
                         rsi_1d_aligned[i] > 50 and 
                         chop_1d_aligned[i] < 61.8)
            
            # Short entry: price below KAMA, RSI < 50, and chop < 61.8 (trending market)
            short_cond = (close[i] < kama_1d_aligned[i] and 
                          rsi_1d_aligned[i] < 50 and 
                          chop_1d_aligned[i] < 61.8)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or RSI < 40
            if close[i] < kama_1d_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or RSI > 60
            if close[i] > kama_1d_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA trend filter with RSI momentum and Choppiness Index regime filter on 12h timeframe.
# Uses 1d KAMA as trend filter (price above/below KAMA), 1d RSI for momentum (>50/<50),
# and 1d Choppiness Index to avoid ranging markets (chop < 61.8 = trending).
# Long when price > KAMA, RSI > 50, and trending market; short when price < KAMA, RSI < 50, and trending.
# Exits when price crosses KAMA or RSI reaches extreme levels.
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Works in bull markets (trend following) and bear markets (avoids whipsaws via chop filter).