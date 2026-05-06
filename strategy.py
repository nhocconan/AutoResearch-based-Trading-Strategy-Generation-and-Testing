#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day KAMA trend direction with volume confirmation and chop filter
# Long when 1-day KAMA is rising, price > KAMA, volume > 1.3x average, and chop > 61.8 (range)
# Short when 1-day KAMA is falling, price < KAMA, volume > 1.3x average, and chop > 61.8 (range)
# Uses daily KAMA for trend, volume for confirmation, and chop to avoid whipsaw in trends
# Designed to work in range-bound markets (2025+) via mean reversion at trend extremes
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "12h_1dKAMA20_Volume_Chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day KAMA (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Efficiency Ratio for KAMA
    change = abs(df_1d['close'].diff(10)).values
    volatility = df_1d['close'].diff().abs().rolling(window=10, min_periods=1).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(df_1d['close'].values)
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(kama)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.where(np.diff(kama, prepend=kama[0]) > 0, 1, -1)
    
    # Align KAMA and direction to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Choppiness Index (14-period) on 1-day data
    atr_14 = pd.DataFrame({'high': df_1d['high'], 'low': df_1d['low'], 'close': df_1d['close']})
    atr_14['tr1'] = atr_14['high'] - atr_14['low']
    atr_14['tr2'] = abs(atr_14['high'] - atr_14['close'].shift(1))
    atr_14['tr3'] = abs(atr_14['low'] - atr_14['close'].shift(1))
    atr_14['tr'] = atr_14[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr_14_val = atr_14['tr'].rolling(window=14, min_periods=14).mean().values
    
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max().values
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14_val * 14 / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid div by zero
    
    # Align chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(kama_dir_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising, price > KAMA, volume confirmation, chop > 61.8 (range)
            if (kama_dir_aligned[i] == 1 and close[i] > kama_aligned[i] and 
                volume_filter[i] and chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, price < KAMA, volume confirmation, chop > 61.8 (range)
            elif (kama_dir_aligned[i] == -1 and close[i] < kama_aligned[i] and 
                  volume_filter[i] and chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falling or chop < 38.2 (trend) or price < KAMA
            if (kama_dir_aligned[i] == -1 or chop_aligned[i] < 38.2 or close[i] < kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rising or chop < 38.2 (trend) or price > KAMA
            if (kama_dir_aligned[i] == 1 or chop_aligned[i] < 38.2 or close[i] > kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals