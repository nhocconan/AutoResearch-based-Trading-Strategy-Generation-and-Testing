#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_filter_v1
# Strategy: Daily KAMA trend with RSI momentum and weekly chop regime filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, capturing true trends while avoiding whipsaws.
# RSI adds momentum confirmation to avoid counter-trend entries.
# Weekly chop filter identifies ranging markets (CHOP > 61.8) where we avoid trend trades,
# and trending markets (CHOP < 38.2) where we follow KAMA direction.
# Works in bull markets by catching trends early and in bear markets by avoiding false signals
# during consolidations and capturing genuine breakdowns.
# Uses tight entry conditions to limit trades (~15-25/year) and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_filter_v1"
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
    
    # Load weekly data ONCE before loop for chop filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly chop regime: EHLERS CHOPPINESS INDEX (14)
    # Higher values = more ranging/choppy, lower values = more trending
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Pad first element
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14) - sum of true ranges
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_raw = np.where(range_14 > 0, 100 * np.log10(tr / range_14) / np.log10(14), 100)
    chop = pd.Series(chop_raw).fillna(50).values  # Neutral when undefined
    
    # Chop regime thresholds
    chop_trending = chop < 38.2   # Trending market
    chop_ranging = chop > 61.8    # Ranging/choppy market
    
    # Load daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio: abs(net change) / sum of absolute changes
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = pd.Series(np.abs(np.diff(close_1d, prepend=close_1d[0]))).rolling(window=10, min_periods=10).sum().values
    net_change = np.abs(close_1d - np.roll(close_1d, 1))
    net_change[0] = 0
    er = np.where(volatility > 0, net_change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Fast=2, Slow=30
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Daily RSI(14) for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)  # Neutral when undefined
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(chop[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA trend direction
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # RSI momentum filters
        rsi_bullish = rsi_aligned[i] > 55  # Avoid weak momentum
        rsi_bearish = rsi_aligned[i] < 45  # Avoid weak momentum
        
        # Chop regime: only trade in trending markets, avoid ranging
        is_trending = chop_trending[i]
        is_ranging = chop_ranging[i]
        
        # Entry logic: KAMA trend + RSI momentum + trending regime
        if price_above_kama and rsi_bullish and is_trending and position != 1:
            position = 1
            signals[i] = 0.25
        elif price_below_kama and rsi_bearish and is_trending and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite KAMA cross OR entering ranging market
        elif position == 1 and (price_below_kama or is_ranging):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (price_above_kama or is_ranging):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals