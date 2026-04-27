#!/usr/bin/env python3
"""
#100776 - 12h_KAMA_Trend_RSI_MeanReversion_1dChop_Filter
Hypothesis: KAMA trend following on 12h with RSI mean-reversion entries and 1d chop filter. Works in bull (trend following) and bear (mean reversion in range). Targets 12-37 trades/year to stay within limits.
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
    
    # Get 1d data for chop filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA on 12h for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # KAMA calculation
    def calculate_kama(close_series, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close_series, n=1))
        volatility = np.abs(np.diff(close_series, n=1))
        er = np.zeros_like(close_series)
        for i in range(1, len(close_series)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        er = pd.Series(er).rolling(window=length, min_periods=length).mean().values
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close_series)
        kama[0] = close_series[0]
        for i in range(1, len(close_series)):
            kama[i] = kama[i-1] + sc[i] * (close_series[i] - kama[i-1])
        return kama
    
    kama_12h = calculate_kama(close_12h, length=10, fast=2, slow=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 1d RSI for mean reversion
    def calculate_rsi(close_series, length=14):
        delta = np.diff(close_series)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, length=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Choppiness Index
    def calculate_chop(high_series, low_series, close_series, length=14):
        atr = np.zeros_like(close_series)
        for i in range(1, len(close_series)):
            tr = max(high_series[i] - low_series[i], 
                     abs(high_series[i] - close_series[i-1]),
                     abs(low_series[i] - close_series[i-1]))
            atr[i] = tr
        atr_sum = pd.Series(atr).rolling(window=length, min_periods=length).sum().values
        hh = pd.Series(high_series).rolling(window=length, min_periods=length).max().values
        ll = pd.Series(low_series).rolling(window=length, min_periods=length).min().values
        chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(length)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, length=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below KAMA
        trend_up = close[i] > kama_12h_aligned[i]
        trend_down = close[i] < kama_12h_aligned[i]
        
        # Mean reversion signals from RSI
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Chop filter: only trade in ranging markets (chop > 50)
        in_range = chop_1d_aligned[i] > 50
        
        # Long: trend up + RSI oversold + in range
        if trend_up and rsi_oversold and in_range:
            signals[i] = 0.25
            position = 1
        # Short: trend down + RSI overbought + in range
        elif trend_down and rsi_overbought and in_range:
            signals[i] = -0.25
            position = -1
        # Exit: RSI returns to neutral
        elif position == 1 and rsi_1d_aligned[i] > 50:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi_1d_aligned[i] < 50:
            signals[i] = 0.0
            position = 0
        # Hold
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_Trend_RSI_MeanReversion_1dChop_Filter"
timeframe = "12h"
leverage = 1.0