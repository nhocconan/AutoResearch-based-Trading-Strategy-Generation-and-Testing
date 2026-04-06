#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction.
RSI filters for momentum exhaustion. Chop filter avoids whipsaws in ranging markets.
Works in bull (trend following) and bear (mean reversion in chop).
Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i-1]):
            kama[i] = kama[i-1] + sc[i-1] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # pad first value
    rsi = np.concatenate([[50], rsi])
    
    # Chop Chopiness Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max/Min close over period
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_close - min_close) != 0,
                    100 * np.log10(atr_sum / (max_close - min_close)) / np.log10(14),
                    50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 30  # For KAMA, RSI, Chop
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: trend change or RSI overbought
            if (close[i] < kama[i] or
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend change or RSI oversold
            if (close[i] > kama[i] or
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: KAMA cross with RSI filter and chop regime
            # Only trade when chop is high (ranging market) for mean reversion
            # or when trend is aligned
            kama_cross_up = close[i] > kama[i] and close[i-1] <= kama[i-1]
            kama_cross_down = close[i] < kama[i] and close[i-1] >= kama[i-1]
            
            # Weekly trend filter: only go with trend in low chop, against in high chop
            weekly_uptrend = close[i] > ema_20_1w_aligned[i]
            
            if chop[i] > 61.8:  # high chop - mean reversion
                # Mean reversion: fade KAMA crosses
                if kama_cross_down and weekly_uptrend and rsi[i] > 50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif kama_cross_up and not weekly_uptrend and rsi[i] < 50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:  # low chop - trend following
                # Trend following: go with KAMA crosses
                if kama_cross_up and rsi[i] < 70:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif kama_cross_down and rsi[i] > 30:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            # Default: hold
            if position == 0 and signals[i] == 0.0:
                signals[i] = 0.0
    
    return signals