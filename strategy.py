#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v1
1d strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI for momentum and Choppiness Index for regime filtering.
- Long: KAMA upward + RSI > 50 + Chop < 61.8 (trending market)
- Short: KAMA downward + RSI < 50 + Chop < 61.8 (trending market)
- Exit: Opposite KAMA direction or Chop > 61.8 (choppy/ranging market)
Designed for ~10-25 trades/year per symbol (40-100 total over 4 years)
Uses weekly trend filter to avoid counter-trend trades in strong trends
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
    
    # ========== KAMA Calculation (ER=10, Fast=2, Slow=30) ==========
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    vol = np.concatenate([np.full(10, np.nan), vol[9:]])  # align
    er = np.where(vol != 0, change / vol, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    # KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ========== RSI(14) ==========
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # ========== Choppiness Index (14) ==========
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    # Sum of TR over 14 periods
    atr_sum = np.full(n, np.nan)
    for i in range(13, n):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    # Highest high and lowest low over 14 periods
    max_high = np.full(n, np.nan)
    min_low = np.full(n, np.nan)
    for i in range(13, n):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    # Chop calculation
    chop = np.full(n, np.nan)
    for i in range(13, n):
        if atr_sum[i] > 0:
            chop[i] = 100 * np.log10(max_high[i] - min_low[i]) / np.log10(14) / np.log10(atr_sum[i])
        else:
            chop[i] = 50
    
    # ========== Weekly Trend Filter (Higher Timeframe) ==========
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA(34) for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # ========== Signals ==========
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure all indicators are valid
    
    for i in range(start_idx, n):
        # Skip if any data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction (trend)
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI condition (momentum)
        rsi_bull = rsi[i] > 50
        rsi_bear = rsi[i] < 50
        
        # Chop regime (trending vs ranging)
        trending = chop[i] < 61.8
        
        # Weekly trend filter
        weekly_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
        weekly_downtrend = ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + trending + weekly uptrend
            if kama_up and rsi_bull and trending and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + trending + weekly downtrend
            elif kama_down and rsi_bear and trending and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down OR choppy market OR weekly trend change
            if not kama_up or not trending or not weekly_uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up OR choppy market OR weekly trend change
            if not kama_down or not trending or not weekly_downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0