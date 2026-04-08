#!/usr/bin/env python3
"""
12h KAMA Direction with 1d RSI and Chop Filter
Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely,
in ranging markets it stays flat. Combined with 1d RSI for overbought/oversold and
Choppiness Index to filter ranging markets, this captures trends while avoiding whipsaws.
Works in bull/bear by only taking signals aligned with higher timeframe trend.
Target: 20-30 trades/year.
"""
name = "12h_kama_1d_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for filters - call ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (12h)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 1d RSI (14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    # Prepend first value
    rsi_1d = np.concatenate([[50], rsi_1d])
    
    # Calculate 1d Choppiness Index (14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Sum of ATR
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max/Min range
    max_h = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_h - min_l
    # Chop
    chop = 100 * np.log10(atr_sum / range_max_min) / np.log10(14)
    chop = np.where(range_max_min != 0, chop, 50)
    
    # Volume spike detector: current volume > 1.5 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Get aligned 1d values for current 12h bar
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI filters: avoid extremes
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        
        # Chop filter: only trade when NOT choppy (trending market)
        not_choppy = chop_aligned[i] < 61.8  # Below chop threshold = trending
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR chop increases
            if not price_above_kama or chop_aligned[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR chop increases
            if not price_below_kama or chop_aligned[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume spike and aligned conditions
            if (volume_spike[i] and price_above_kama and 
                rsi_not_overbought and not_choppy):
                position = 1
                signals[i] = 0.25
            elif (volume_spike[i] and price_below_kama and 
                  rsi_not_oversold and not_choppy):
                position = -1
                signals[i] = -0.25
    
    return signals