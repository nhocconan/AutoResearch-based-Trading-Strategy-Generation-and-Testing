#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d EMA34 trend filter and 1d KAMA direction for entries.
# Long when 1d EMA34 up, KAMA rising, and price > KAMA.
# Short when 1d EMA34 down, KAMA falling, and price < KAMA.
# Includes volatility filter (ATR ratio) to avoid choppy markets and reduce false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.
# Works in bull (trend follow) and bear (trend still exists in downtrends).

name = "12h_1dEMA34_KAMA"
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
    
    # Get 1d data for trend filter, KAMA, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = ema_34_1d > np.roll(ema_34_1d, 1)
    trend_1d_up = np.where(np.isnan(trend_1d_up), False, trend_1d_up)
    
    # 1d KAMA (using ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # This is incorrect, need to compute properly
    # Recompute ER correctly
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i < 10:
            er[i] = np.nan
        else:
            direction = np.abs(close_1d[i] - close_1d[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = direction / volatility_sum if volatility_sum != 0 else 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_rising = kama > np.roll(kama, 1)
    kama_rising = np.where(np.isnan(kama_rising), False, kama_rising)
    
    # 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    kama_rising_aligned = align_htf_to_ltf(prices, df_1d, kama_rising.astype(float))
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volatility filter: ATR ratio (current ATR / 50-period average ATR)
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(kama_aligned[i]) or
            np.isnan(kama_rising_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d EMA34 up, KAMA rising, price > KAMA, ATR ratio < 1.2 (low volatility)
            if (trend_1d_up_aligned[i] and
                kama_rising_aligned[i] and
                close[i] > kama_aligned[i] and
                atr_ratio_aligned[i] < 1.2):
                signals[i] = 0.25
                position = 1
            # Short: 1d EMA34 down, KAMA falling, price < KAMA, ATR ratio < 1.2 (low volatility)
            elif (not trend_1d_up_aligned[i] and
                  not kama_rising_aligned[i] and
                  close[i] < kama_aligned[i] and
                  atr_ratio_aligned[i] < 1.2):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend break, KAMA falling, or price < KAMA
            if (not trend_1d_up_aligned[i] or
                not kama_rising_aligned[i] or
                close[i] < kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend break, KAMA rising, or price > KAMA
            if (trend_1d_up_aligned[i] or
                kama_rising_aligned[i] or
                close[i] > kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals