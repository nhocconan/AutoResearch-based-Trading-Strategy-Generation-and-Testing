#!/usr/bin/env python3
"""
4h_KAMA_Direction_VolumeSpike_ChopFilter
Hypothesis: 4h strategy using Kaufman Adaptive Moving Average (KAMA) direction with volume spike confirmation and choppiness regime filter. 
KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends. Volume spike confirms breakout strength. 
Choppiness filter avoids trading in extreme ranging conditions. Designed for low trade frequency (target: 20-50/year) to minimize fee drag.
Works in both bull and bear markets by aligning with adaptive trend and avoiding false signals in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d KAMA for trend filter (adaptive to market conditions)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # KAMA(10,2,30) - fast=10, slow=30
    close_1d_series = pd.Series(close_1d)
    change = abs(close_1d_series - close_1d_series.shift(10))  # 10-period net change
    volatility = abs(close_1d_series.diff()).rolling(window=10, min_periods=10).sum()  # 10-period volatility
    er = change / volatility.replace(0, np.nan)  # Efficiency Ratio
    er = er.fillna(0)  # Handle division by zero
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Smoothing Constant
    kama_1d = pd.Series(close_1d).ewm(alpha=sc, adjust=False).mean().values
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1h volume average for spike detection (more responsive than 4h)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    volume_1h = df_1h['volume'].values
    vol_avg_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_avg_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_avg_1h)
    
    # Calculate 4h choppiness index for regime filter
    chop_window = 14
    true_range = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    true_range[0] = high[0] - low[0]  # First period TR
    atr = pd.Series(true_range).rolling(window=chop_window, min_periods=chop_window).mean().values
    max_high = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    min_low = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    chop = 100 * np.log10(atr * chop_window / (max_high - min_low)) / np.log10(chop_window)
    # Handle division by zero and invalid values
    chop = np.where((max_high - min_low) == 0, 50, chop)  # Neutral when no range
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for KAMA, volume average, and chop
    start_idx = max(100, 30, 20, chop_window)
    
    size = 0.25  # 25% position size
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(vol_avg_1h_aligned[i]) or 
            np.isnan(chop[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        vol_spike = volume[i] > (2.0 * vol_avg_1h_aligned[i])
        chop_val = chop[i]
        
        # Choppiness regime: avoid extreme chop (both too high and too low can be problematic)
        # CHOP > 61.8 = ranging (chop), CHOP < 38.2 = trending
        # We avoid extreme ranging (CHOP > 70) and extreme trending (CHOP < 30) to reduce whipsaws
        regime_filter = (chop_val >= 30) and (chop_val <= 70)
        
        if position == 0:
            # Flat - look for entry: KAMA direction with volume spike and regime filter
            # Long: price above KAMA AND volume spike AND regime filter
            # Short: price below KAMA AND volume spike AND regime filter
            if close_val > kama_val and vol_spike and regime_filter:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif close_val < kama_val and vol_spike and regime_filter:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price crosses below KAMA (trend change) or regime becomes extreme
            if close_val < kama_val or not regime_filter:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price crosses above KAMA (trend change) or regime becomes extreme
            if close_val > kama_val or not regime_filter:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Direction_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0