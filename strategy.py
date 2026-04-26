#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filtering.
Only trade when KAMA slope aligns with RSI > 50 (long) or < 50 (short) AND market is trending (CHOP < 38.2).
Avoids whipsaws in ranging markets. Discrete sizing 0.25 targets 30-100 trades over 4 years.
Weekly EMA50 filter ensures we only trade in alignment with higher timeframe trend.
Works in bull via trend continuation and in bear by avoiding counter-trend trades via weekly EMA filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    # ER = Efficiency Ratio, SC = Smoothing Constant
    # KAMA adapts to market noise: faster in trends, slower in ranges
    def calculate_kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    # KAMA slope: positive = rising trend, negative = falling trend
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (CHOP) ===
    # Measures whether market is trending (low CHOP) or ranging (high CHOP)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    chop = np.where((max_high - min_low) != 0,
                    100 * np.log10(np.sum(atr, axis=1) / (max_high - min_low)) / np.log10(14),
                    50)
    # Fix for first values
    chop[:13] = 50
    
    # === Weekly EMA50 for HTF trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 30 for KAMA/RSI/CHOP warmup)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_slope[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long conditions: KAMA rising, RSI > 50, trending market (CHOP < 38.2), price above weekly EMA50
        long_condition = (kama_slope[i] > 0) and (rsi[i] > 50) and (chop[i] < 38.2) and (close[i] > ema_50_1w_aligned[i])
        # Short conditions: KAMA falling, RSI < 50, trending market (CHOP < 38.2), price below weekly EMA50
        short_condition = (kama_slope[i] < 0) and (rsi[i] < 50) and (chop[i] < 38.2) and (close[i] < ema_50_1w_aligned[i])
        
        # Exit conditions: trend change or ranging market
        exit_long = (kama_slope[i] <= 0) or (rsi[i] <= 50) or (chop[i] >= 38.2)
        exit_short = (kama_slope[i] >= 0) or (rsi[i] >= 50) or (chop[i] >= 38.2)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0