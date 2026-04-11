#!/usr/bin/env python3
# 4h_1d_kama_rsi_chop_v1
# Strategy: 4-hour Kaufman Adaptive Moving Average (KAMA) with RSI momentum and Choppiness Index regime filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, providing reliable trend signals. RSI (14) confirms momentum strength.
# Choppiness Index (CHOP) filters range-bound markets (CHOP > 61.8) to avoid false signals.
# Long: KAMA bullish crossover + RSI > 50 + CHOP < 61.8 (trending market)
# Short: KAMA bearish crossover + RSI < 50 + CHOP < 61.8 (trending market)
# Uses tight entry conditions to limit trades (~20-40/year) and avoid fee drag. Works in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_rsi_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR (14)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Sum of TRUE ranges over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    # Max(high) - Min(low) over 14 periods
    max_h_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_l_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    range_14 = max_h_14 - min_l_14
    
    # Choppiness Index: 100 * log10(sum_tr_14 / range_14) / log10(14)
    chop_raw = 100 * np.log10(sum_tr_14 / range_14) / np.log10(14)
    chop = chop_raw.fillna(50).values  # neutral when undefined
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h KAMA (10, 2, 30) - ER = 2, FAST = 2, SLOW = 30
    close_series = pd.Series(close)
    change = np.abs(close_series.diff(10))  # 10-period net change
    volatility = close_series.diff().abs().rolling(window=10, min_periods=10).sum()  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)  # Efficiency Ratio
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # Smoothing Constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 4h RSI (14)
    delta = close_series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(rsi[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA crossovers
        kama_cross_up = kama[i-1] <= close_series.iloc[i-1] and kama[i] > close_series.iloc[i]
        kama_cross_down = kama[i-1] >= close_series.iloc[i-1] and kama[i] < close_series.iloc[i]
        
        # RSI momentum filter
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_market = chop_1d_aligned[i] < 61.8
        
        # Entry logic: KAMA crossover + RSI momentum + trending regime
        if kama_cross_up and rsi_bullish and trending_market and position != 1:
            position = 1
            signals[i] = 0.25
        elif kama_cross_down and rsi_bearish and trending_market and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite KAMA crossover
        elif position == 1 and kama_cross_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and kama_cross_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals