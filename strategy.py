#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and Choppiness index regime filter
# Long when price touches S3 pivot level in trending market (CHOP < 38.2) with volume > 1.5x 12h average
# Short when price touches R3 pivot level in trending market (CHOP < 38.2) with volume > 1.5x 12h average
# Exit when price reaches opposite pivot level (S1/R1) or trend changes (CHOP > 61.8)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_camarilla_1d_chop_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 6)
    # S2 = C - (Range * 1.1 / 4)
    # S3 = C - (Range * 1.1 * 2 / 6)
    # R3 = C + (Range * 1.1 * 2 / 6)
    # R2 = C + (Range * 1.1 / 4)
    # R1 = C + (Range * 1.1 / 6)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    s3 = close_1d - (range_hl * 1.1 * 2 / 6)
    s1 = close_1d - (range_hl * 1.1 / 6)
    r1 = close_1d + (range_hl * 1.1 / 6)
    r3 = close_1d + (range_hl * 1.1 * 2 / 6)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # 12h data for volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # Choppiness index for regime filter (using 12h data)
    # CHOP = 100 * log10(sum(TR over n) / (ATR * n)) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_period = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum / (atr_period * 14)) / np.log10(14)
    
    # ATR(14) for stoploss
    atr = atr_period  # already calculated above
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(s3_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(volume_ma_12h_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches S1 (take profit) or chop > 61.8 (ranging market)
            elif close[i] <= s1_aligned[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches R1 (take profit) or chop > 61.8 (ranging market)
            elif close[i] >= r1_aligned[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trending market (CHOP < 38.2)
            # Long: price touches S3 level, volume spike, trending market
            if (close[i] <= s3_aligned[i] and
                volume[i] > 1.5 * volume_ma_12h_aligned[i] and
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches R3 level, volume spike, trending market
            elif (close[i] >= r3_aligned[i] and
                  volume[i] > 1.5 * volume_ma_12h_aligned[i] and
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals