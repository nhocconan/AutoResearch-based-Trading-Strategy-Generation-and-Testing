#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R1/S1 breakout with 1d volume spike and chop regime filter
# Enter long when price breaks above R1 with volume > 2.0x 24-period average and chop > 61.8 (ranging)
# Enter short when price breaks below S1 with volume > 2.0x 24-period average and chop > 61.8 (ranging)
# Exit on opposite pivot level touch (S1 for long, R1 for short) or chop < 38.2 (trending)
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in ranging markets via mean reversion at extreme pivot levels with volume confirmation.
# Uses 1d HTF for pivot calculation and volume average, aligned to 12h LTF.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Camarilla pivots, volume average, and chop ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === Camarilla Pivot Levels (R1, S1) ===
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d Volume Confirmation (24-period average) ===
    vol_ma_24 = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_24)
    
    # === Choppiness Index (14-period) ===
    # CHOP = 100 * log10(sum(ATR14) / (n * (HHV - LLV))) / log10(n)
    # Where ATR14 = ATR(14), HHV = highest high, LLV = lowest low over n periods
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    sum_atr14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hhv_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    llv_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = hhv_14 - llv_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop_1d = 100 * np.log10(sum_atr14 / (14 * range_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 2.0  # 2.0x average volume
        chop_val = chop_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price touches S1 or chop < 38.2 (trending market)
            if price <= s1_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches R1 or chop < 38.2 (trending market)
            if price >= r1_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above R1 AND volume confirmation AND chop > 61.8 (ranging)
            if price > r1_val and vol_confirm and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price breaks below S1 AND volume confirmation AND chop > 61.8 (ranging)
            elif price < s1_val and vol_confirm and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_1dVolConfirm_ChopRegime"
timeframe = "12h"
leverage = 1.0