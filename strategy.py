#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v2
# Strategy: Daily KAMA with RSI filter and weekly Choppiness Index regime filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market efficiency, RSI identifies overbought/oversold conditions,
# and Choppiness Index filters trending vs ranging markets. Works in both bull and bear markets
# by adapting to volatility regimes and avoiding false signals in chop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(atr14) / (hh - ll)) / log10(14)
    sum_atr14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr14 / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) == 0, 50, chop)  # Avoid division by zero
    chop = np.where(np.isnan(chop), 50, chop)  # Fill NaN with neutral value
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Daily KAMA (10-period ER, 2/30 fast/slow SC)
    er_period = 10
    fast_sc = 2
    slow_sc = 30
    
    change = np.abs(close - np.roll(close, er_period))
    change = np.concatenate([[np.nan] * er_period, change[er_period:]])  # First er_period values NaN
    
    diff = np.abs(np.diff(close, prepend=close[0]))
    sum_diff = pd.Series(diff).rolling(window=er_period, min_periods=er_period).sum().values
    
    er = np.where(sum_diff != 0, change / sum_diff, 0)
    sc = (er * (fast_sc / er_period - slow_sc / er_period) + slow_sc / er_period) ** 2
    
    kama = np.full_like(close, np.nan)
    kama[er_period] = close[er_period]  # Seed
    
    for i in range(er_period + 1, len(close)):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Daily RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # When no loss, RSI=100
    rsi = np.where(avg_gain == 0, 0, rsi)    # When no gain, RSI=0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending
        chop_val = chop_aligned[i]
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        # Entry logic: In ranging markets, mean revert at RSI extremes
        if is_ranging and rsi_oversold and price_above_kama and position != 1:
            position = 1
            signals[i] = 0.25
        elif is_ranging and rsi_overbought and price_below_kama and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: RSI returns to neutral or chop indicates trending
        elif position == 1 and (rsi[i] > 50 or not is_ranging):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 50 or not is_ranging):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals