#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filtering.
Enter long when KAMA up, RSI > 50, and CHOP < 38.2 (trending market).
Enter short when KAMA down, RSI < 50, and CHOP < 38.2.
Exit when opposite signal or CHOP > 61.8 (choppy regime).
Uses discrete position sizing (0.25) to minimize fee drag. Designed for low trade frequency
(7-25 trades/year) to work in both bull and bear markets by adapting to regime.
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
    
    # Get 1w data for HTF trend filter (optional, can be removed if too restrictive)
    # df_1w = get_htf_data(prices, '1w')
    # if len(df_1w) < 10:
    #     return np.zeros(n)
    
    # Calculate KAMA on close
    # Efficiency ratio: abs(close - close[10]) / sum(abs(diff)) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of abs daily changes
    # Pad arrays for alignment
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), volatility])
    # Avoid division by zero
    er = np.divide(change_padded, volatility_padded, out=np.full_like(change_padded, np.nan), where=volatility_padded!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # Calculate Choppiness Index(14)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sumTR / (HH - LL)) / log10(14)
    hh_ll = hh - ll
    # Avoid division by zero and log of zero
    ratio = np.divide(tr_sum, hh_ll, out=np.full_like(tr_sum, np.nan), where=(hh_ll!=0) & (~np.isnan(hh_ll)))
    log_ratio = np.log10(ratio)
    chop = 100 * log_ratio / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of KAMA seed(10), RSI(14), Chop(14)
    start_idx = max(10, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade in trending market (CHOP < 38.2)
        trending_market = chop[i] < 38.2
        choppy_market = chop[i] > 61.8
        
        if position == 0:
            # Long: KAMA up (close > kama), RSI > 50, trending market
            long_signal = (close[i] > kama[i]) and (rsi[i] > 50) and trending_market
            # Short: KAMA down (close < kama), RSI < 50, trending market
            short_signal = (close[i] < kama[i]) and (rsi[i] < 50) and trending_market
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA down OR RSI < 50 OR choppy regime
            if (close[i] < kama[i]) or (rsi[i] < 50) or choppy_market:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA up OR RSI > 50 OR choppy regime
            if (close[i] > kama[i]) or (rsi[i] > 50) or choppy_market:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0