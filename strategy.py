#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Chop_Filter
Hypothesis: KAMA adapts to market efficiency, providing smooth trend direction. 
Combined with RSI for momentum and Choppiness Index for regime filtering, this creates 
a robust strategy that works in both trending and ranging markets. 
Designed for ~20-40 trades/year to minimize fee drag on 4h timeframe.
"""

name = "4h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 0 else np.abs(np.diff(close)).sum()
    # For 1D arrays, calculate rolling volatility
    volatility_rolling = pd.Series(close).rolling(window=er_length).sum(np.abs(np.diff(close, prepend=close[0]))).values
    # Fix: Calculate ER properly
    price_change = np.abs(np.diff(close, n=er_length, prepend=close[:er_length]))
    total_change = pd.Series(close).rolling(window=er_length).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    er = np.where(total_change != 0, price_change / total_change, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, length=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_choppiness(high, low, close, length=14):
    """Calculate Choppiness Index"""
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
    
    max_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
    min_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
    
    chop = np.where(atr != 0, 100 * np.log10((max_high - min_low) / (atr * length)) / np.log10(length), 50)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA on close prices
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    
    # Calculate RSI
    rsi = calculate_rsi(close, length=14)
    
    # Calculate Choppiness Index
    chop = calculate_choppiness(high, low, close, length=14)
    
    # 1d trend filter: close vs KAMA on 1d
    kama_1d = calculate_kama(df_1d['close'].values, er_length=10, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Warmup period
        # Determine market regime: chop < 38.2 = trending, chop > 61.8 = ranging
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        if position == 0:
            # LONG: KAMA up + RSI > 50 + volume + (trending OR ranging with mean reversion)
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                volume_filter[i] and
                ((is_trending and close[i] > kama[i]) or 
                 (is_ranging and rsi[i] < 40))):  # Mean reversion in ranging market
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down + RSI < 50 + volume + (trending OR ranging with mean reversion)
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  volume_filter[i] and
                  ((is_trending and close[i] < kama[i]) or 
                   (is_ranging and rsi[i] > 60))):  # Mean reversion in ranging market
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down OR RSI < 40 (overbought mean reversion)
            if (kama[i] < kama[i-1] or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up OR RSI > 60 (oversold mean reversion)
            if (kama[i] > kama[i-1] or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals