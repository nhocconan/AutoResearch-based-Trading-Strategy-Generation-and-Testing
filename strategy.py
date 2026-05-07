#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA direction + 1d RSI + 1d Choppiness filter.
# Long when KAMA is rising (bullish trend), RSI < 40 (oversold), and Choppiness > 61.8 (ranging market).
# Short when KAMA is falling (bearish trend), RSI > 60 (overbought), and Choppiness > 61.8 (ranging market).
# Exit when KAMA direction changes or RSI returns to neutral (40-60).
# This strategy captures mean reversion in ranging markets with trend alignment, avoiding strong trends.
# The 1d Choppiness filter ensures we only trade in ranging conditions where mean reversion works best.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by adapting to ranging conditions that occur in all regimes.

name = "4h_KAMA_RSI_Choppiness"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (14, 2, 30) - Kaufman Adaptive Moving Average
    def calculate_kama(price, er_len=10, fast_ma=2, slow_ma=30):
        change = np.abs(np.diff(price, prepend=price[0]))
        vol = np.sum(np.abs(np.diff(price, prepend=price[0])), axis=0) if len(price.shape) > 1 else np.abs(np.diff(price, prepend=price[0]))
        if len(price.shape) > 1:
            vol = np.sum(vol, axis=1)
        er = np.where(vol != 0, change / vol, 0)
        sc = (er * (2/(fast_ma+1) - 2/(slow_ma+1)) + 2/(slow_ma+1)) ** 2
        kama = np.zeros_like(price)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    kama_rising = np.zeros_like(kama, dtype=bool)
    kama_falling = np.zeros_like(kama, dtype=bool)
    kama_rising[1:] = kama[1:] > kama[:-1]
    kama_falling[1:] = kama[1:] < kama[:-1]
    
    # 1d RSI(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([np.full(14, np.nan), rsi_1d[14:]])  # warmup period
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d Choppiness Index(14)
    def calculate_choppiness(high, low, close, cp_len=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(np.diff(high, prepend=high[0]))
        tr2 = np.abs(np.diff(low, prepend=low[0]))
        tr3 = np.abs(np.diff(close, prepend=close[0]))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = pd.Series(tr).ewm(alpha=1/cp_len, adjust=False).mean().values
        atr = np.concatenate([np.full(cp_len, np.nan), atr[cp_len:]])  # warmup period
        
        highest_high = pd.Series(high).rolling(window=cp_len, min_periods=cp_len).max().values
        lowest_low = pd.Series(low).rolling(window=cp_len, min_periods=cp_len).min().values
        sum_atr = pd.Series(atr).rolling(window=cp_len, min_periods=cp_len).sum().values
        
        chop = np.where((highest_high - lowest_low) != 0, 
                        100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(cp_len), 
                        50)
        return chop
    
    chop_1d = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Sufficient warmup for KAMA and other indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(kama_rising[i]) or np.isnan(kama_falling[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA rising, RSI < 40 (oversold), Choppiness > 61.8 (ranging)
            long_cond = kama_rising[i] and (rsi_1d_aligned[i] < 40) and (chop_1d_aligned[i] > 61.8)
            # Short conditions: KAMA falling, RSI > 60 (overbought), Choppiness > 61.8 (ranging)
            short_cond = kama_falling[i] and (rsi_1d_aligned[i] > 60) and (chop_1d_aligned[i] > 61.8)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA direction changes or RSI returns to neutral (40-60)
            if not kama_rising[i] or (rsi_1d_aligned[i] >= 40 and rsi_1d_aligned[i] <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA direction changes or RSI returns to neutral (40-60)
            if not kama_falling[i] or (rsi_1d_aligned[i] >= 40 and rsi_1d_aligned[i] <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals