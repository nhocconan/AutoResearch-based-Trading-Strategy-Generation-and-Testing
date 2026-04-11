#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop
Strategy: 1-day KAMA direction + RSI filter + weekly chop regime filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses Kaufman's Adaptive Moving Average (KAMA) to identify adaptive trend direction,
combined with RSI for momentum confirmation and weekly Choppiness Index to filter ranging markets.
Designed for low trade frequency (<25/year) to minimize fee decay while capturing trends in
both bull and bear markets via adaptive trend following. Target: 15-20 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1-day KAMA (adaptive trend) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1-day RSI (momentum) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1-week Choppiness Index (regime filter) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero and log of zero
    range_14 = hh_14 - ll_14
    chop = np.where((range_14 > 0) & (sum_tr_14 > 0),
                    100 * np.log10(sum_tr_14 / range_14) / np.log10(14),
                    50)  # neutral when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend direction: price relative to KAMA
        above_kama = price_close > kama[i]
        below_kama = price_close < kama[i]
        
        # RSI filters: avoid extremes, look for momentum
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Chop filter: trending market (Chop < 38.2) OR ranging market (Chop > 61.8)
        # We'll use trending filter for momentum entries
        trending_market = chop_aligned[i] < 38.2
        
        # Long conditions: price above KAMA + RSI bullish + trending market
        long_signal = above_kama and rsi_bullish and trending_market
        
        # Short conditions: price below KAMA + RSI bearish + trending market
        short_signal = below_kama and rsi_bearish and trending_market
        
        # Exit when price crosses KAMA in opposite direction
        exit_long = position == 1 and below_kama
        exit_short = position == -1 and above_kama
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals