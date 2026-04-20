#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume confirmation and 12h Choppiness regime filter
# - Long when price breaks above Donchian(20) high AND 12h volume > 1.5x 20-period average AND 12h Choppiness > 61.8 (ranging market)
# - Short when price breaks below Donchian(20) low AND 12h volume > 1.5x 20-period average AND 12h Choppiness > 61.8
# - Exit when price returns to Donchian midpoint or volatility regime shifts
# - Designed for 4h timeframe with volume and regime filters to reduce false breakouts
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for volume and Choppiness filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_avg_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Choppiness Index (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_12h
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_atr_14 / (hh_14 - ll_14)) / np.log10(14)
    chop = np.where((hh_14 - ll_14) == 0, 50, chop)  # avoid division by zero
    
    # Align 12h indicators to 4h timeframe
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate Donchian channels on 4h (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_avg_20_aligned[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol_avg = vol_avg_20_aligned[i]
        chop_val = chop_aligned[i]
        vol_12h_current = df_12h['volume'].values[len(df_12h) - len(prices) + i // 3] if i // 3 < len(df_12h) else vol_avg
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume confirmation + ranging market (chop > 61.8)
            if price > donch_high[i] and vol_12h_current > 1.5 * vol_avg and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume confirmation + ranging market (chop > 61.8)
            elif price < donch_low[i] and vol_12h_current > 1.5 * vol_avg and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian midpoint OR chop drops below 38.2 (trending market)
            if price < donch_mid[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian midpoint OR chop drops below 38.2 (trending market)
            if price > donch_mid[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0