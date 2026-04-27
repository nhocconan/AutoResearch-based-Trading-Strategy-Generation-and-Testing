#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 1d timeframe for trend direction, combined with RSI(14) for momentum confirmation and Choppiness Index for regime filtering. Enters long when 1d close > KAMA AND RSI > 50 AND choppy regime filter passed (CHOP < 61.8 = trending). Enters short when 1d close < KAMA AND RSI < 50 AND choppy regime filter passed. Uses 1d timeframe to minimize fee drag, targeting 15-30 trades/year. KAMA adapts to market noise, reducing false signals in ranging markets, while RSI confirms momentum and chop filter avoids whipsaws. Works in both bull and bear markets via adaptive trend detection and regime filter.
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
    
    # Get 1d data for KAMA, RSI, and chop calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA on 1d close
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d))
    change = np.concatenate([[0], change])  # align length
    volatility = np.abs(np.diff(close_1d))
    volatility = np.concatenate([[0], volatility])  # align length
    
    er_num = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    er_den = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = er_num / (er_den + 1e-10)
    
    # Smoothing constants
    fastest = 2 / (2 + 1)  # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d)
    delta = np.concatenate([[0], delta])  # align length
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Choppiness Index filter: avoid breakouts in choppy markets (CHOP > 61.8 = range)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[0], tr1])  # align length
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1 * 14 / np.log10(14) / (max_high - min_low + 1e-10))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_filter = chop_aligned < 61.8  # True = trending market, allow trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need KAMA (10), RSI (14), chop (14)
    start_idx = max(10, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_ok = chop_filter[i]
        
        if position == 0:
            # Look for entry: price above/below KAMA with RSI confirmation and trending regime
            # Long: close > KAMA AND RSI > 50 AND trending market
            long_condition = (close_val > kama_val) and (rsi_val > 50) and chop_ok
            # Short: close < KAMA AND RSI < 50 AND trending market
            short_condition = (close_val < kama_val) and (rsi_val < 50) and chop_ok
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price crosses below KAMA OR RSI < 50
            exit_condition = (close_val <= kama_val) or (rsi_val < 50)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price crosses above KAMA OR RSI > 50
            exit_condition = (close_val >= kama_val) or (rsi_val > 50)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0