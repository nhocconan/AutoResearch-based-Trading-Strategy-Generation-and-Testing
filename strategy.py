#!/usr/bin/env python3
"""
1d_kama_rsi_chop_v1
Hypothesis: KAMA identifies trend direction on daily timeframe, RSI provides overbought/oversold signals,
and Choppiness Index filters ranging vs trending markets. In ranging markets (Chop > 61.8), we mean-revert
at RSI extremes. In trending markets (Chop < 38.2), we follow KAMA direction. Works in both bull and bear
markets by adapting to market regime. Targets 7-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Efficiency Ratio
    er = np.zeros_like(close)
    for i in range(er_period, len(close)):
        er[i] = np.abs(close[i] - close[i-er_period]) / np.sum(volatility[i-er_period+1:i+1])
    
    # Smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index"""
    atr = np.zeros(len(high))
    atr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    for i in range(1, len(atr)):
        atr[i] = (atr[i-1] * (period-1) + atr[i]) / period
    
    hh = np.zeros(len(high))
    ll = np.zeros(len(high))
    hh[0] = high[0]
    ll[0] = low[0]
    for i in range(1, len(high)):
        hh[i] = max(high[i], hh[i-1])
        ll[i] = min(low[i], ll[i-1])
    
    chop = np.zeros(len(high))
    for i in range(period-1, len(high)):
        sum_atr = 0
        for j in range(i-period+1, i+1):
            sum_atr += atr[j]
        if hh[i] != ll[i]:
            chop[i] = 100 * np.log10(sum_atr / (hh[i] - ll[i])) / np.log10(period)
        else:
            chop[i] = 50
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA for trend direction
    kama = calculate_kama(close, er_period=10, fast=2, slow=30)
    
    # RSI for overbought/oversold
    rsi = calculate_rsi(close, period=14)
    
    # Weekly data for Choppiness (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    chop_1w = calculate_choppiness(high_1w, low_1w, close_1w, period=14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        ranging = chop_aligned[i] > 61.8  # Chop > 61.8 = ranging (mean revert)
        trending = chop_aligned[i] < 38.2  # Chop < 38.2 = trending (trend follow)
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or chop shifts to trending against position
            if rsi[i] > 70 or (trending and kama[i] < close[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or chop shifts to trending against position
            if rsi[i] < 30 or (trending and kama[i] > close[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long in ranging market: RSI < 30 (oversold)
            if ranging and rsi[i] < 30:
                position = 1
                signals[i] = 0.25
            # Short in ranging market: RSI > 70 (overbought)
            elif ranging and rsi[i] > 70:
                position = -1
                signals[i] = -0.25
            # Long in trending market: price above KAMA
            elif trending and close[i] > kama[i]:
                position = 1
                signals[i] = 0.25
            # Short in trending market: price below KAMA
            elif trending and close[i] < kama[i]:
                position = -1
                signals[i] = -0.25
    
    return signals