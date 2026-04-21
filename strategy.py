# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA direction + RSI + Chop regime filter
- KAMA adapts to market noise, reducing whipsaw in choppy markets
- RSI identifies overbought/oversold conditions for mean reversion
- Chop filter ensures we only trade in ranging markets (CHOP > 61.8)
- Works in both bull and bear by adapting to volatility regime
- Target: 10-25 trades/year to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily data for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / np.where(volatility[1:] == 0, 1, volatility[1:])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / np.where(avg_loss == 0, 1, avg_loss))
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Chop calculation
    atr1 = np.maximum(high_1d - low_1d,
                      np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                 np.abs(low_1d - np.roll(close_1d, 1))))
    atr1[0] = high_1d[0] - low_1d[0]
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.where(max_high - min_low == 0, 1, max_high - min_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Align daily data to 1d timeframe (no change needed)
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(ema200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema200_1w_val = ema200_1w_aligned[i]
        
        # Chop filter: only trade in ranging markets (CHOP > 61.8)
        ranging = chop_val > 61.8
        
        if position == 0:
            # Long: price > KAMA, RSI < 40 (oversold), ranging market
            if price > kama_val and rsi_val < 40 and ranging:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI > 60 (overbought), ranging market
            elif price < kama_val and rsi_val > 60 and ranging:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on KAMA cross down or RSI > 70
                if price < kama_val or rsi_val > 70:
                    exit_signal = True
            elif position == -1:  # short position
                # Exit on KAMA cross up or RSI < 30
                if price > kama_val or rsi_val < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Range"
timeframe = "1d"
leverage = 1.0