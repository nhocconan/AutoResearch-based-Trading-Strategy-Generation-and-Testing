#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Daily Pivot + Volume Spike + Chop Regime Filter
# Daily pivot levels provide clear support/resistance zones from prior day's action
# Volume spike confirms institutional interest at key levels
# Chop filter avoids false breakouts in ranging markets (CHOP > 61.8) and
# enables trend-following breakouts in trending markets (CHOP < 38.2)
# Works in bull markets (breakouts continue trends) and bear markets (fades false breakouts in ranges)
# Target: 20-40 trades/year with clear, high-probability setups

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day's data)
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    r1 = 2 * pivot - low_1d[:-1]
    s1 = 2 * pivot - high_1d[:-1]
    r2 = pivot + (high_1d[:-1] - low_1d[:-1])
    s2 = pivot - (high_1d[:-1] - low_1d[:-1])
    
    # Align pivot levels to 4h timeframe (using prior day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 4h ATR(14) for volatility
    tr1 = np.maximum(high[1:], low[:-1]) - np.minimum(high[1:], low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h Choppiness Index(14)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    
    # Calculate 4h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            continue
        
        # Volume spike condition (2x average volume)
        volume_spike = volume[i] > 2 * vol_ma[i]
        
        # Long conditions: price breaks above resistance with volume
        long_breakout = (close[i] > r1_aligned[i] or close[i] > r2_aligned[i]) and volume_spike
        # Long fade in range: price at support with volume
        long_fade = (close[i] <= s1_aligned[i] * 1.001 and close[i] >= s1_aligned[i] * 0.999) and volume_spike and chop[i] > 61.8
        
        # Short conditions: price breaks below support with volume
        short_breakout = (close[i] < s1_aligned[i] or close[i] < s2_aligned[i]) and volume_spike
        # Short fade in range: price at resistance with volume
        short_fade = (close[i] >= r1_aligned[i] * 0.999 and close[i] <= r1_aligned[i] * 1.001) and volume_spike and chop[i] > 61.8
        
        # Enter long
        if (long_breakout or long_fade) and position <= 0:
            position = 1
            signals[i] = position_size
        # Enter short
        elif (short_breakout or short_fade) and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit when price returns to pivot zone
        elif position == 1 and (close[i] <= pivot_aligned[i] * 1.001 and close[i] >= pivot_aligned[i] * 0.999):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= pivot_aligned[i] * 1.001 and close[i] >= pivot_aligned[i] * 0.999):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_DailyPivot_Volume_Chop_Filter"
timeframe = "4h"
leverage = 1.0