#!/usr/bin/env python3
name = "12h_KAMA_Direction_RSI_1dChop_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    # Efficiency ratio (ER)
    change = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'].iloc[0]))
    volatility = np.abs(np.diff(df_1d['close']))
    er = change / (volatility.rolling(window=10, min_periods=10).sum().values + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(df_1d['close'])
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # Calculate RSI(14) on daily close
    delta = np.diff(df_1d['close'], prepend=df_1d['close'].iloc[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP) on daily data
    atr = np.zeros(len(df_1d))
    tr1 = np.abs(np.diff(df_1d['high'], prepend=df_1d['high'].iloc[0]))
    tr2 = np.abs(np.diff(df_1d['low'], prepend=df_1d['low'].iloc[0]))
    tr3 = np.abs(df_1d['high'] - df_1d['low'].shift(1).fillna(df_1d['low'].iloc[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr * 14) / (max_high - min_low + 1e-10)) / np.log10(14)
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: only trade when market is trending (CHOP < 38.2) or ranging (CHOP > 61.8)
        # In trending markets, follow KAMA direction
        # In ranging markets, mean revert at RSI extremes
        if chop_aligned[i] < 38.2:  # Trending market
            if position == 0:
                if close[i] > kama_aligned[i] and rsi_aligned[i] > 50:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                if close[i] < kama_aligned[i] or rsi_aligned[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] > kama_aligned[i] or rsi_aligned[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        elif chop_aligned[i] > 61.8:  # Ranging market
            if position == 0:
                if rsi_aligned[i] < 30 and close[i] > kama_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif rsi_aligned[i] > 70 and close[i] < kama_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                if rsi_aligned[i] > 50 or close[i] < kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if rsi_aligned[i] < 50 or close[i] > kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:  # Neutral chop zone - no trading
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

# Hypothesis: KAMA direction + RSI + Chop regime filter on 12h timeframe
# KAMA adapts to market efficiency - follows price closely in trends, stays flat in ranges
# Chop regime filter identifies market state: <38.2 = trending, >61.8 = ranging
# In trending markets: follow KAMA direction with RSI confirmation (avoid false breaks)
# In ranging markets: mean revert at RSI extremes (30/70) with KAMA as dynamic support/resistance
# Works in both bull (follow trend) and bear (mean revert in ranges) markets
# Position size 0.25 limits risk while maintaining sufficient trade frequency (~20-40/year)