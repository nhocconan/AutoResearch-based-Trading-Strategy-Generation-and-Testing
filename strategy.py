#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum confirmation and Choppiness Index(14) as regime filter.
Enter long when KAMA trends up, RSI > 50, and market is trending (CHOP < 38.2).
Enter short when KAMA trends down, RSI < 50, and market is trending (CHOP < 38.2).
Exit on opposite signal or when market becomes choppy (CHOP > 61.8).
Designed for 1d timeframe with ~10-25 trades/year, avoiding overtrading via strict regime filter.
Works in both bull and bear markets by adapting to trending regimes only.
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
    
    # Weekly data for HTF trend filter (optional reinforcement)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend confirmation
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))  # sum of |diff| over 10 periods
    er = np.zeros(n)
    er[10:] = change[10:] / volatility[10:]
    er[volatility == 0] = 0
    # Smoothing constants: sc = [ER*(fastest - slowest) + slowest]^2
    fastest = 2.0 / (2 + 1)
    slowest = 2.0 / (30 + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Initialize KAMA
    kama = np.zeros(n)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = np.nan  # not enough data
    
    # Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(14)
    tr1 = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.zeros(n)
    chop[:13] = np.nan
    valid = (max_high - min_low) > 0
    chop[14:] = 100 * np.log10(sum_atr1[14:] / (max_high[14:] - min_low[14:])) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(14, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop[i] < 38.2
        is_choppy = chop[i] > 61.8
        
        if position == 0:
            # Long: KAMA trending up, RSI > 50, trending market
            kama_up = kama[i] > kama[i-1]
            long_setup = kama_up and (rsi[i] > 50) and is_trending
            # Short: KAMA trending down, RSI < 50, trending market
            kama_down = kama[i] < kama[i-1]
            short_setup = kama_down and (rsi[i] < 50) and is_trending
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: KAMA turns down OR RSI < 50 OR market becomes choppy
            if (kama[i] < kama[i-1]) or (rsi[i] < 50) or is_choppy:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: KAMA turns up OR RSI > 50 OR market becomes choppy
            if (kama[i] > kama[i-1]) or (rsi[i] > 50) or is_choppy:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0