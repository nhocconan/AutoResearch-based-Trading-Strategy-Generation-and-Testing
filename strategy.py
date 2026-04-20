#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + Volume Spike + Choppiness Filter
# - Long when price breaks above Donchian(20) high + volume > 2x average + chop > 61.8 (range)
# - Short when price breaks below Donchian(20) low + volume > 2x average + chop > 61.8 (range)
# - Uses Donchian for structure, volume for confirmation, chop to avoid trends
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for choppiness index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR) for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR(14) and Sum of TR for 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Calculate Choppiness Index (CHOP) for 14 periods
    # CHOP = 100 * log10(sum(tr14) / (atr14 * 14)) / log10(14)
    chop_raw = 100 * np.log10(tr_sum_14 / (atr_14 * 14)) / np.log10(14)
    chop = chop_raw  # Already in 0-100 scale
    
    # Align 1d Choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian Channel (20) on 4h
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume (20-period) on 4h
    volume = prices['volume'].values
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(avg_volume[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Long entry: price > Donchian high + volume spike + chop > 61.8 (range)
            if price > donch_high[i] and vol > 2 * avg_volume[i] and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian low + volume spike + chop > 61.8 (range)
            elif price < donch_low[i] and vol > 2 * avg_volume[i] and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < Donchian low or chop < 38.2 (trend)
            if price < donch_low[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > Donchian high or chop < 38.2 (trend)
            if price > donch_high[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0