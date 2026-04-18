#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter
Hypothesis: Kaufman's Adaptive Moving Average (KAMA) adapts to market noise,
reducing whipsaws in ranging markets. Combined with RSI extremes and Choppiness
Index regime filter, this strategy captures strong trends while avoiding false
signals in chop. Works in both bull (trend following) and bear (mean reversion
in chop) markets by adapting to market conditions.
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle first 10 elements
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start KAMA at index 9
    for i in range(10, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first element
    rsi = np.concatenate([np.array([np.nan]), rsi])
    
    # Choppiness Index (14-period)
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    # First element
    tr[0] = atr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high - lowest_low) != 0,
                    100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14),
                    50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        trend = ema34_1w_aligned[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        # Market regime: chop > 61.8 = range (mean revert), chop < 38.2 = trending (trend follow)
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Enter long in trending market: price > KAMA + RSI > 50
            # Enter short in trending market: price < KAMA + RSI < 50
            # In ranging market: mean reversion at RSI extremes
            if is_trending:
                if close[i] > kama_val and rsi_val > 50:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama_val and rsi_val < 50:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Mean reversion: buy oversold, sell overbought
                if rsi_val < 30:
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 70:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: trend change or RSI overbought in range
            if (is_trending and close[i] < kama_val) or (is_ranging and rsi_val > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend change or RSI oversold in range
            if (is_trending and close[i] > kama_val) or (is_ranging and rsi_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0