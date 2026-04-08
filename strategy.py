#!/usr/bin/env python3
"""
12h_1d_kama_rsi_volatility_breakout_v1
Hypothesis: KAMA trend direction on 12h combined with RSI overbought/oversold and volatility breakout.
- Long when KAMA rising, RSI < 30, and price breaks above 1d Donchian high with volume confirmation
- Short when KAMA falling, RSI > 70, and price breaks below 1d Donchian low with volume confirmation
- Uses 1d volatility (ATR) to filter breakouts
- Designed for low trade frequency (15-25/year) to minimize fee drag
- Works in bull/bear via KAMA trend filter and volatility breakout logic
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_rsi_volatility_breakout_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    if len(close) < er_length:
        return np.full_like(close, np.nan, dtype=float)
    
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama = np.full_like(close, np.nan, dtype=float)
    kama[er_length] = close[er_length]
    for i in range(er_length + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full_like(high, np.nan, dtype=float)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Calculate 1d ATR (14-period) for volatility filter
    atr_1d = calculate_atr(high_1d, low_1d, df_1d['close'].values, 14)
    
    # Calculate KAMA on 12h (10,2,30)
    kama = calculate_kama(close, 10, 2, 30)
    
    # Calculate RSI on 12h (14-period)
    rsi = calculate_rsi(close, 14)
    
    # Align indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: 20-period average on 12h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        atr_val = atr_1d_aligned[i]
        kama_rising = kama[i] > kama[i-1] if i > 0 else False
        kama_falling = kama[i] < kama[i-1] if i > 0 else False
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 1:  # Long
            # Exit: price closes below Donchian low or volatility contracts
            if price < lower or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price closes above Donchian high or volatility contracts
            if price > upper or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: KAMA rising, RSI oversold, break above Donchian high with volume confirmation
            if kama_rising and rsi_oversold and price > upper and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: KAMA falling, RSI overbought, break below Donchian low with volume confirmation
            elif kama_falling and rsi_overbought and price < lower and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals