#!/usr/bin/env python3
"""
1d_KAMA_Regime_Filter_Volume_ATRStop_v1
Hypothesis: 1d KAMA (adaptive trend) with choppiness regime filter and volume confirmation.
In trending markets (CHOP < 38.2): follow KAMA direction (long above, short below).
In ranging markets (CHOP > 61.8): no entries to avoid whipsaw.
Volume confirmation (1.5x average) filters false signals.
ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to limit fee drag and drawdown.
Designed to work in both bull and bear markets by avoiding range-bound whipsaw.
Timeframe: 1d, uses 1w HTF for trend context (optional enhancement).
Target: 30-100 total trades over 4 years = 7-25/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === KAMA (adaptive trend) ===
    close = prices['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of abs changes
    # Avoid division by zero
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Choppiness Index (CHOP) ===
    high = prices['high'].values
    low = prices['low'].values
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero and log(0)
    chop = np.full_like(close, np.nan)
    mask = (max_hh - min_ll) > 0
    chop[mask] = 100 * np.log10(sum_tr[mask] / (max_hh[mask] - min_ll[mask])) / np.log10(14)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Only enter in trending markets (CHOP < 38.2) with volume confirmation
            if chop[i] < 38.2 and volume_confirmed:
                if price > kama[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price < kama[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Filter_Volume_ATRStop_v1"
timeframe = "1d"
leverage = 1.0