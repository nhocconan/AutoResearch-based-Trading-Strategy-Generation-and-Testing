#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness index regime filter combined with 1d EMA trend and volume spike.
# Long when: 4h Choppiness > 61.8 (range) AND price > 1d EMA50 AND volume > 1.5x 20-period average.
# Short when: 4h Choppiness > 61.8 (range) AND price < 1d EMA50 AND volume > 1.5x 20-period average.
# Exit when Choppiness < 38.2 (trending) or opposite condition met.
# This strategy mean-reverts in ranging markets (high chop) while avoiding trending markets (low chop).
# Works in both bull and bear markets by using 1d EMA for direction and volume for confirmation.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "4h_Chop_EMA50_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Choppiness index (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range14 = highest_high - lowest_low
    range14 = np.where(range14 == 0, 1e-10, range14)
    
    chop = 100 * np.log10(atr14 * 14 / range14) / np.log10(14)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(chop[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: high chop (range), price above EMA50, volume filter
            long_cond = (chop[i] > 61.8) and (close[i] > ema50_1d_aligned[i]) and volume_filter[i]
            # Short conditions: high chop (range), price below EMA50, volume filter
            short_cond = (chop[i] > 61.8) and (close[i] < ema50_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: chop drops below 38.2 (trending) or price crosses below EMA50
            if chop[i] < 38.2 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: chop drops below 38.2 (trending) or price crosses above EMA50
            if chop[i] < 38.2 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals