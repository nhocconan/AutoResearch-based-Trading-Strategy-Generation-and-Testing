#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, combined with RSI for momentum confirmation and Choppiness Index for regime filtering. Enter long when KAMA turns up, RSI > 50, and market is trending (CHOP < 38.2). Enter short when KAMA turns down, RSI < 50, and market is trending. Exit on opposite KAMA signal. This strategy targets 7-25 trades/year to minimize fee drag while capturing sustained trends in both bull and bear markets.
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
    
    # === KAMA Calculation (ER=10, Fast=2, Slow=30) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    # Pad volatility to match change length
    volatility = np.concatenate([np.full(9, np.nan), volatility[9:]]) if len(volatility) > 9 else np.full_like(change, np.nan)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Seed
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # === Choppiness Index (CHOP, 14-period) ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # CHOP formula
    chop = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
    # Avoid division by zero or invalid values
    chop = np.where((hh - ll) > 0, chop, 50.0)
    
    # === Weekly Trend Filter (HTF: 1w) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need sufficient data for all indicators
    start_idx = max(10, 14, 14, 50)  # KAMA seed, RSI, CHOP, 1w EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or
            np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        is_trending = chop[i] < 38.2
        
        if position == 0:
            # Flat - look for KAMA turn with RSI confirmation and weekly trend alignment
            kama_up = kama[i] > kama[i-1]
            kama_down = kama[i] < kama[i-1]
            rsi_bullish = rsi[i] > 50
            rsi_bearish = rsi[i] < 50
            weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
            
            # Long: KAMA turning up, RSI > 50, weekly uptrend, and trending market
            if kama_up and rsi_bullish and weekly_uptrend and is_trending:
                signals[i] = size
                position = 1
            # Short: KAMA turning down, RSI < 50, weekly downtrend, and trending market
            elif kama_down and rsi_bearish and weekly_downtrend and is_trending:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when KAMA turns down (trend change)
            if kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when KAMA turns up (trend change)
            if kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0