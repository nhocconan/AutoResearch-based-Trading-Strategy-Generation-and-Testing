#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Chop_Filter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) direction on 12h timeframe
with RSI momentum filter and Choppiness Index regime filter to avoid whipsaws.
KAMA adapts to market noise, reducing false signals in choppy markets.
RSI filters for momentum strength, and Choppiness Index ensures we only trade
in trending markets (CHOP < 38.2) or mean-revert in ranging markets (CHOP > 61.8).
Target: 15-25 trades/year to stay well under fee drag limits while capturing
strong moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    vol = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 1 else np.abs(change[0])
    # Avoid division by zero
    er = np.where(vol != 0, change / vol, 0)
    # For rolling calculation, we need to compute ER over a window
    er_series = pd.Series(er)
    er_rolled = er_series.rolling(window=er_length, min_periods=1).mean()
    sc = (er_rolled * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_chop(high, low, close, window=14):
    """Calculate Choppiness Index."""
    atr = np.zeros(len(close))
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # True Range sum over window
    tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum()
    
    # Highest high and lowest low over window
    hh = pd.Series(high).rolling(window=window, min_periods=window).max()
    ll = pd.Series(low).rolling(window=window, min_periods=window).min()
    
    # CHOP formula
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(window)
    # Handle division by zero or invalid cases
    chop = np.where((hh - ll) != 0, chop, 50)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA on 1d for trend direction
    kama_1d = calculate_kama(close_1d, er_length=10, fast_sc=2, slow_sc=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI on 12h for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate Choppiness Index on 1d for regime filter
    chop = calculate_chop(high_1d, low_1d, close_1d, window=14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama = kama_1d_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long conditions: price above KAMA (uptrend), RSI > 50 (momentum),
            # CHOP < 38.2 (trending market), volume spike
            if price > kama and rsi_val > 50 and chop_val < 38.2 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price below KAMA (downtrend), RSI < 50,
            # CHOP < 38.2 (trending market), volume spike
            elif price < kama and rsi_val < 50 and chop_val < 38.2 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA OR chop becomes too high (ranging)
            if price < kama or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA OR chop becomes too high (ranging)
            if price > kama or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0