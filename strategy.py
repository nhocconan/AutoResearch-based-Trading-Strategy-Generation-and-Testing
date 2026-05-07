#!/usr/bin/env python3
# 12h_KAMA_Direction_Volume_Chop
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) direction as primary trend filter,
# combined with volume spike confirmation and Choppiness Index regime filter to avoid whipsaws.
# KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets.
# Volume spikes confirm institutional interest at trend continuation points.
# Choppiness Index > 61.8 indicates ranging market (avoid trend following), < 38.2 indicates trending (follow KAMA).
# Designed for 12h timeframe to target 15-25 trades/year per symbol, staying within optimal trade frequency.
# Works in bull markets by following KAMA uptrends, in bear markets by following KAMA downtrends,
# and avoids false signals during high-chop periods.

timeframe = "12h"
name = "12h_KAMA_Direction_Volume_Chop"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        # Fix: volatility calculation needs to be rolling sum
        volatility = pd.Series(close).rolling(window=er_length).sum().values
        volatility = np.where(volatility == 0, 1e-10, volatility)  # Avoid division by zero
        er = change / volatility
        er = np.where(np.isnan(er), 0, er)
        sc = (er * (fast_sc/slow_sc - 1) + 1) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Get daily data for Choppiness Index (needs daily high/low/close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA on 12h close
    kama_12h = kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Choppiness Index on daily timeframe
    def chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First true range
        atr = pd.Series(tr).rolling(window=length, min_periods=length).mean().values
        
        highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        range_hl = np.where(range_hl == 0, 1e-10, range_hl)
        
        chop_val = 100 * np.log10(atr * np.sqrt(length) / range_hl) / np.log10(length)
        return chop_val
    
    chop_1d = chop(high_1d, low_1d, close_1d, length=14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume spike detection: 2x average volume (48-period = 2 days on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(48, 30)  # Ensure we have volume MA and KAMA warmup
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_12h[i]) or np.isnan(chop_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when market is trending (Chop < 38.2) or strongly trending (Chop < 50)
        trending_market = chop_12h_aligned[i] < 50
        
        if position == 0 and trending_market:
            # Long: price above KAMA with volume spike
            if (close[i] > kama_12h[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume spike
            elif (close[i] < kama_12h[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA OR chop becomes too high (ranging market)
            if (close[i] < kama_12h[i] or chop_12h_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA OR chop becomes too high (ranging market)
            if (close[i] > kama_12h[i] or chop_12h_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals