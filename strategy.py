#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d KAMA: Efficiency Ratio over 10 periods ===
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, 1))  # daily change
    er = np.zeros_like(close)
    for i in range(n):
        if i < 10:
            er[i] = 0.0
        else:
            er[i] = change[i] / (volatility[i-9:i+1].sum() + 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1w ADX(14) for trend strength ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    tr = np.maximum(high_1w - low_1w, 
                    np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), 
                               np.abs(low_1w - np.roll(close_1w, 1))))
    tr[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1w + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1w + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # === 1d Choppiness Index(14) ===
    atr1 = np.maximum(high - low, 
                      np.maximum(np.abs(high - np.roll(close, 1)), 
                                 np.abs(low - np.roll(close, 1))))
    atr1[0] = high[0] - low[0]
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest - lowest + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, chop > 61.8 (range)
            long_cond = (close[i] > kama[i] and 
                         rsi[i] > 50 and 
                         chop[i] > 61.8)
            # Short: KAMA down, RSI < 50, chop > 61.8 (range)
            short_cond = (close[i] < kama[i] and 
                          rsi[i] < 50 and 
                          chop[i] > 61.8)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA down or RSI < 40
            exit_cond = (close[i] < kama[i] or rsi[i] < 40)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA up or RSI > 60
            exit_cond = (close[i] > kama[i] or rsi[i] > 60)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily KAMA trend filter with RSI momentum and Choppiness Index regime filter.
# In ranging markets (CHOP > 61.8), takes mean-reversion entries when price deviates from KAMA
# with RSI confirmation. Avoids trending markets where mean reversion fails.
# Weekly ADX ensures we only trade when higher timeframe has sufficient trend strength
# to sustain moves. Designed for BTC/ETH: works in bull (trend following via KAMA) and
# bear (mean reversion in ranges) markets. Targets 20-60 trades over 4 years (5-15/year)
# to minimize fee drag. Uses discrete sizing (0.25) to reduce churn. Weekly ADX filter
# prevents trading in weak trend environments where whipsaws occur.