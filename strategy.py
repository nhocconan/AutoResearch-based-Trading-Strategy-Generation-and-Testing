#!/usr/bin/env python3
# 12h_1d_KAMA_RSI_Chop
# Hypothesis: 12h Kaufman Adaptive Moving Average (KAMA) direction with RSI filter and 1d Choppiness regime. 
# KAMA adapts to market noise, reducing false signals in ranging markets. RSI avoids overbought/oversold extremes. 
# Choppiness index filters for trending regimes (CHOP < 38.2) to avoid whipsaws. Designed for low trade frequency (<30/year) 
# to minimize fee drag in both bull and bear markets. Works on BTC/ETH by combining adaptive trend, momentum, and regime filters.

name = "12h_1d_KAMA_RSI_Chop"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Choppiness index (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h KAMA calculation ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    change = np.concatenate([[np.nan]*10, change])  # align to original index
    
    # Sum of absolute daily changes over 10 periods
    abs_change = np.abs(np.diff(close, n=1))
    abs_change = np.concatenate([[np.nan], abs_change])
    volatility = pd.Series(abs_change).rolling(window=10, min_periods=10).sum().values
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction (slope)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # === 12h RSI (14) ===
    delta = np.diff(close, n=1)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d Choppiness Index (14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chi = np.where(tr_sum != 0, -100 * np.log10(tr_sum / (hh - ll)) / np.log10(14), 50)
    # Handle division by zero or invalid cases
    chi = np.where((hh - ll) == 0, 50, chi)
    chi = np.where(np.isnan(chi), 50, chi)
    
    # Align 1d Choppiness to 12h timeframe
    chi_aligned = align_htf_to_ltf(prices, df_1d, chi)
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: ensure all indicators are valid
    start_idx = max(50, 14, 10)  # RSI(14), KAMA needs lookback
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_slope[i]) or
            np.isnan(rsi[i]) or
            np.isnan(chi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Conditions
        kama_up = kama_slope[i] > 0
        kama_down = kama_slope[i] < 0
        rsi_not_extreme = (rsi[i] > 30) & (rsi[i] < 70)  # avoid overbought/oversold
        trending_market = chi_aligned[i] < 38.2  # chop < 38.2 = trending
        
        if position == 0:
            # Long: KAMA up + RSI not extreme + trending market
            if kama_up and rsi_not_extreme and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI not extreme + trending market
            elif kama_down and rsi_not_extreme and trending_market:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when KAMA reverses OR RSI extreme OR market becomes choppy
            if position == 1:
                if (kama_down or not rsi_not_extreme or chi_aligned[i] >= 38.2):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (kama_up or not rsi_not_extreme or chi_aligned[i] >= 38.2):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals