#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h ATR(14) for volatility normalization
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3_4h = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 4h ATR to 4h index (no shift, ATR is contemporaneous)
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h Donchian channels (20-period)
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 4h index (no shift, breakout uses closed 4h bar)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Get 1d HTF data for regime filter (choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Chopiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3_1d = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / log10(14)
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    chop_14 = np.where(
        (range_14 > 0) & (sum_tr_14 > 0),
        100 * np.log10(sum_tr_14 / range_14) / np.log10(14),
        50  # neutral when range is zero
    )
    
    # Align 1d Chopiness to 4h index
    chop_14_aligned = align_htf_to_ltf(prices, df_1d, chop_14)
    
    # Get 1h data for entry timing (optional refinement)
    # Calculate 1h volume ratio (current vs 50-period average)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_ratio = volume / (vol_ma_50 + 1e-10)
    
    signals = np.zeros(n)
    
    # Start loop after warmup period
    start_idx = max(100, 50)  # ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i]) or np.isnan(chop_14_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Chopiness Index < 38.2 = trending (favor breakouts)
        # Chopiness Index > 61.8 = ranging (avoid breakouts)
        if chop_14_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 4h Donchian upper (20) - use close of 4h bar
        # 2. Volatility filter: ATR > 0.5% of price (avoid extremely low volatility)
        # 3. Volume confirmation: volume > 1.3x average
        if (close[i] > upper_20_aligned[i] and
            atr_14_4h_aligned[i] > 0.005 * close[i] and
            volume_ratio[i] > 1.3):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 4h Donchian lower (20)
        # 2. Volatility filter: ATR > 0.5% of price
        # 3. Volume confirmation: volume > 1.3x average
        elif (close[i] < lower_20_aligned[i] and
              atr_14_4h_aligned[i] > 0.005 * close[i] and
              volume_ratio[i] > 1.3):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_ChopFilter_Volume_ATR"
timeframe = "4h"
leverage = 1.0