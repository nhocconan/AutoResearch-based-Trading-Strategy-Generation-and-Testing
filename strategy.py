#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 4h KAMA trend direction + 1d volume confirmation
# Long when: CHOP > 61.8 (range) AND KAMA rising AND volume > 1.5x 1d average volume
# Short when: CHOP > 61.8 (range) AND KAMA falling AND volume > 1.5x 1d average volume
# Chop filter avoids false trends, KAMA adapts to volatility, volume confirms conviction
# Target: 50-150 total trades over 4 years (12-38/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Choppiness Index (14-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR14
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_14 = hh - ll
    chop = 100 * np.log10(tr_sum / range_14) / np.log10(14)
    chop[range_14 == 0] = 100  # avoid division by zero
    chop[np.isnan(chop)] = 50  # neutral value for warmup
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    # === 4h KAMA (adaptive moving average) ===
    # Efficiency Ratio
    change = np.abs(close_4h - np.roll(close_4h, 10))
    change[0] = 0
    volatility = np.sum(np.abs(np.diff(close_4h, prepend=close_4h[0])), axis=0)
    # Simplified: rolling sum of absolute changes
    volatility = pd.Series(np.abs(np.diff(close_4h, prepend=close_4h[0]))).rolling(window=10, min_periods=1).sum().values
    er = change / volatility
    er[volatility == 0] = 0
    er[np.isnan(er)] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # === 1d Volume Confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmrow = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmrow, n):
        # Skip if any data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(kama_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        kama_val = kama_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        
        # Chop filter: range-bound market (CHOP > 61.8)
        range_filter = chop_val > 61.8
        
        # KAMA direction: rising or falling
        kama_rising = kama_val > kama_aligned[i-1] if i > 0 else False
        kama_falling = kama_val < kama_aligned[i-1] if i > 0 else False
        
        # Volume confirmation: current volume > 1.5x 1d average volume
        vol_confirm = volume[i] > vol_ma_val * 1.5
        
        # === ENTRY LOGIC ===
        if position == 0 and range_filter and vol_confirm:
            # Long when KAMA rising
            if kama_rising:
                signals[i] = 0.25
                position = 1
                continue
            # Short when KAMA falling
            elif kama_falling:
                signals[i] = -0.25
                position = -1
                continue
        
        # === EXIT LOGIC: reverse signal when conditions fail ===
        if position == 1:
            if not (kama_rising and range_filter and vol_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if not (kama_falling and range_filter and vol_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Choppiness61.8_KAMADir_1dVol1.5x"
timeframe = "4h"
leverage = 1.0