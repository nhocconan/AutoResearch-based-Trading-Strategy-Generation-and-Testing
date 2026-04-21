#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_1w_Volume_Filter
Hypothesis: Daily trend following using Kaufman Adaptive Moving Average (KAMA) with weekly volume confirmation. 
KAMA adapts to market noise, reducing whipsaws in choppy markets. Weekly volume filter ensures trades align with 
institutional participation. Works in both bull and bear by taking long when price > KAMA and volume confirms, 
short when price < KAMA and volume confirms. Targets 10-20 trades/year with strict entry conditions to minimize 
fee drag. Uses 1d ATR for volatility-based stoploss to manage risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, er_length))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[:len(close)], axis=0) if len(close) >= er_length else np.zeros_like(close)
    # Fix volatility calculation
    volatility = np.zeros_like(close)
    for i in range(er_length, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_length:i+1])))
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly average volume
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Load daily data once for KAMA and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily KAMA for trend
    close_1d = df_1d['close'].values
    kama_1d = calculate_kama(close_1d, er_length=10, fast_sc=2, slow_sc=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate daily ATR for volatility filter and stop
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current daily volume > 1.5 * weekly average volume
        volume_ok = volume > 1.5 * vol_ma_1w_aligned[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_1d_aligned[i] > 0.01 * price  # Avoid near-zero ATR
        
        if position == 0:
            # Long: price above KAMA + volume confirmation
            if price > kama_1d_aligned[i] and volume_ok and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below KAMA + volume confirmation
            elif price < kama_1d_aligned[i] and volume_ok and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price below KAMA or stoploss hit
            if price < kama_1d_aligned[i] or price < entry_price - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA or stoploss hit
            if price > kama_1d_aligned[i] or price > entry_price + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_1w_Volume_Filter"
timeframe = "1d"
leverage = 1.0