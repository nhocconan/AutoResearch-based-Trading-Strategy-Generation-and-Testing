#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Regime_v1
Hypothesis: Daily KAMA trend direction combined with RSI extremes and Choppiness Index regime filter.
Works in both bull/bear markets by using KAMA for adaptive trend, RSI(14)<30 or >70 for mean-reversion entries,
and Choppiness Index > 61.8 to identify ranging conditions where mean reversion is effective.
ATR-based stoploss manages risk. Targets 7-25 trades/year to minimize fee drag.
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)   # 2-period EMA smoothing constant
    slow_sc = 2 / (30 + 1)  # 30-period EMA smoothing constant
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0]))
    volatility = np.abs(np.diff(df_1d['close'].values)).rolling(window=10, min_periods=1).sum()
    er = change / volatility
    er = np.where(volatility == 0, 0, er)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(df_1d['close'].values)
    kama[0] = df_1d['close'].values[0]
    for i in range(1, len(kama)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].values[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (no extra delay needed for trend)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d timeframe
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index on 1w timeframe (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1w['high'].values[1:] - df_1w['low'].values[1:]
    tr2 = np.abs(df_1w['high'].values[1:] - df_1w['close'].values[:-1])
    tr3 = np.abs(df_1w['low'].values[1:] - df_1w['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    # ATR for stoploss (14-period) on 1d timeframe
    tr1_d = high[1:] - low[1:]
    tr2_d = np.abs(high[1:] - close[:-1])
    tr3_d = np.abs(low[1:] - close[:-1])
    tr_d = np.concatenate([[np.nan], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(30, 14, 14)  # KAMA, RSI, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(atr_d[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        atr_val = atr_d[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for mean reentry in ranging market (Chop > 61.8)
            # Long: RSI < 30 (oversold) + price > KAMA (bullish bias) + Chop > 61.8 (ranging)
            long_entry = (rsi_val < 30) and (close_val > kama_val) and (chop_val > 61.8)
            # Short: RSI > 70 (overbought) + price < KAMA (bearish bias) + Chop > 61.8 (ranging)
            short_entry = (rsi_val > 70) and (close_val < kama_val) and (chop_val > 61.8)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on RSI > 50 (mean reversion complete) or ATR stoploss
            exit_condition = (rsi_val > 50) or (close_val < entry_price - 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on RSI < 50 (mean reversion complete) or ATR stoploss
            exit_condition = (rsi_val < 50) or (close_val > entry_price + 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0