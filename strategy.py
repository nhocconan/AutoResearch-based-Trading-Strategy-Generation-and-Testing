#!/usr/bin/env python3
"""
12h KAMA Trend with Volume Confirmation and Daily ATR Filter
- KAMA (Kaufman Adaptive Moving Average) adapts to market noise, reducing whipsaws
- Long when price > KAMA, short when price < KAMA
- Volume filter: require volume > 1.5x 20-period average
- Daily ATR filter: only trade when daily ATR > 0.5 * 20-period average of daily ATR (avoid low volatility)
- Designed for 12h timeframe: target 50-150 trades over 4 years
- Works in both bull (trend following) and bear (adaptive filtering reduces false signals)
"""

from typing import Tuple
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_trend_volume_filter_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_ma=2, slow_ma=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 1 else np.abs(change[0])
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility
    sc = (er * (2/(fast_ma+1) - 2/(slow_ma+1)) + 2/(slow_ma+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[1:14])  # Seed with first 14 values
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # 20-period average of daily ATR
    atr_ma_20 = np.zeros_like(atr_14)
    for i in range(len(atr_14)):
        if i >= 19:
            atr_ma_20[i] = np.mean(atr_14[i-19:i+1])
        else:
            atr_ma_20[i] = np.nan
    
    # Daily ATR filter: current ATR > 0.5 * 20-period average ATR
    atr_filter_1d = atr_14 > 0.5 * atr_ma_20
    
    # Align daily ATR filter to 12h timeframe
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter_1d.astype(float))
    
    # Calculate KAMA on 12h close prices
    kama = calculate_kama(close, er_period=10, fast_ma=2, slow_ma=30)
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_filter_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # ATR filter: only trade when volatility is sufficient
        atr_filter = bool(atr_filter_aligned[i])
        
        # KAMA trend signals
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Entry conditions: require both volume and ATR filters
        long_entry = price_above_kama and volume_filter and atr_filter
        short_entry = price_below_kama and volume_filter and atr_filter
        
        # Exit conditions: reverse signal
        long_exit = price_below_kama
        short_exit = price_above_kama
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals