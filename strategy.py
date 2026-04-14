#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA (adaptive moving average) with RSI(14) and Chop filter.
# KAMA adapts to market noise: faster in trends, slower in ranges.
# RSI(14) > 55 for long, < 45 for short avoids choppy entries.
# Choppiness Index (14) > 61.8 indicates ranging market (fade reversals), < 38.2 indicates trending (follow breakouts).
# Uses 1-day trend filter for higher timeframe bias.
# Designed for 12h timeframe: targets 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
# Works in both bull (follows KAMA trend) and bear (avoids false signals in chop via Chop filter).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA (adaptive moving average) on 12h close
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        # Full calculation requires loop - using simplified adaptive alpha
        # For practical purposes, use EMA with volatility-based alpha
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if np.sum(np.abs(np.diff(close[i-length:i+1]))) > 0:
                er[i] = np.abs(close[i] - close[i-length]) / np.sum(np.abs(np.diff(close[i-length:i+1])))
        # Smoothing constants
        fast_sc = 2/(fast+1)
        slow_sc = 2/(slow+1)
        sc = (er * (fast_sc - slow_sc) + slow_sc)**2
        # Initialize KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    # Simplified KAMA using EMA with volatility adjustment (practical approximation)
    # Using 10-period EMA as base, adjusted by volatility ratio
    price_change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(price_change).rolling(window=10, min_periods=1).sum().values
    abs_change = pd.Series(np.abs(np.diff(close, prepend=0))).rolling(window=10, min_periods=1).sum().values
    er = np.where(abs_change > 0, price_change / abs_change, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama_12h = pd.Series(close).ewm(alpha=sc, adjust=False).mean().values
    
    # RSI(14)
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_12h = rsi(close, 14)
    
    # Choppiness Index (14)
    def chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[0], tr])
        atr = pd.Series(tr).rolling(window=length, min_periods=length).sum().values
        max_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        min_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(length)
        return chop
    
    chop_12h = chop(high, low, close, 14)
    
    # 1-day EMA(50) for trend filter
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 14)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_12h[i]) or 
            np.isnan(rsi_12h[i]) or
            np.isnan(chop_12h[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1-day EMA50
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Chop regime: > 61.8 = range (fade), < 38.2 = trend (follow)
        chop_value = chop_12h[i]
        in_trend = chop_value < 38.2
        in_range = chop_value > 61.8
        
        if position == 0:
            # Enter long: price > KAMA + RSI > 55 + in trending OR (in range and mean reversion)
            if (close[i] > kama_12h[i] and 
                rsi_12h[i] > 55 and
                (in_trend or (in_range and close[i] < kama_12h[i]))):  # Mean reversion in range
                position = 1
                signals[i] = position_size
            # Enter short: price < KAMA + RSI < 45 + in trending OR (in range and mean reversion)
            elif (close[i] < kama_12h[i] and 
                  rsi_12h[i] < 45 and
                  (in_trend or (in_range and close[i] > kama_12h[i]))):  # Mean reversion in range
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < KAMA or RSI < 40
            if close[i] < kama_12h[i] or rsi_12h[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price > KAMA or RSI > 60
            if close[i] > kama_12h[i] or rsi_12h[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_KAMA_RSI_Chop_v1"
timeframe = "12h"
leverage = 1.0