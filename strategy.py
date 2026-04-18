#!/usr/bin/env python3
"""
12h_KAMA_RSI_Chop_v1
Strategy: 12h KAMA trend direction + RSI momentum + Choppiness regime filter.
Long: KAMA bullish + RSI > 50 + Chop > 61.8 (range) for mean reversion.
Short: KAMA bearish + RSI < 50 + Chop > 61.8 (range) for mean reversion.
Uses 1D trend filter (EMA50 > EMA200) to avoid counter-trend trades.
Designed for low-frequency, high-conviction trades in ranging markets.
Target: ~20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === KAMA Calculation (12h) ====
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) on 12h ====
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(13, np.nan), rsi])  # align length
    
    # === Choppiness Index (14) on 12h ====
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) != 0,
                    -100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14),
                    50)
    # Fix: proper chop calculation
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    range_hhll = max_high - min_low
    chop = np.where(range_hhll != 0,
                    100 * np.log10(sum_atr / range_hhll) / np.log10(14),
                    50)
    
    # === Daily Trend Filter (EMA50 > EMA200) ====
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all daily and 12h indicators
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA is already 12h, but align for safety
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)   # RSI is 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop) # Chop is 12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200 and indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade with daily trend
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # KAMA direction
        kama_up = close[i] > kama_aligned[i]
        kama_down = close[i] < kama_aligned[i]
        
        # RSI conditions
        rsi_over = rsi_aligned[i] > 50
        rsi_under = rsi_aligned[i] < 50
        
        # Choppiness regime: range-bound market (good for mean reversion)
        chop_high = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: uptrend + KAMA up + RSI > 50 + choppy market
            if uptrend and kama_up and rsi_over and chop_high:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + KAMA down + RSI < 50 + choppy market
            elif downtrend and kama_down and rsi_under and chop_high:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or KAMA down or RSI < 40 or chop low (trending)
            if (not uptrend) or (not kama_up) or (rsi_aligned[i] < 40) or (chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or KAMA up or RSI > 60 or chop low (trending)
            if (not downtrend) or (not kama_down) or (rsi_aligned[i] > 60) or (chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_Chop_v1"
timeframe = "12h"
leverage = 1.0