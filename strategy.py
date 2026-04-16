#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d choppiness regime filter.
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period average AND 1d chop > 61.8 (range).
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period average AND 1d chop > 61.8 (range).
# Exit when price crosses Donchian(20) midline OR chop < 38.2 (trend regime).
# Uses discrete position size 0.30. Volume confirmation reduces false breakouts.
# Choppiness regime filter ensures trading only in ranging markets where mean reversion works.
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in ranging markets (2025 BTC/ETH bear/range) by fading breakouts that fail in chop.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data once before loop for choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h Indicators: Donchian(20) channels ===
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    # Donchian midline = (high + low) / 2
    from pandas import Series
    donchian_high_4h = Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
    
    # === 4h Indicators: Volume confirmation ===
    # Volume > 1.5x 20-period average
    vol_ma_4h = Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_4h  # current volume / average volume
    
    # === 1d Indicators: Choppiness Index (14) ===
    # Chop = 100 * log10(sum(ATR, 14) / (log10(highest high - lowest low, 14)) / log10(14))
    # Simplified: Chop = 100 * log10(ATR_sum / (HH-LL) * log10(14)) / log10(14)
    # We'll use a proxy: Chop > 61.8 = range, Chop < 38.2 = trend
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.maximum(tr1, np.abs(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = high_1d[0] - low_1d[0]
    atr14_1d = Series(tr2).rolling(window=14, min_periods=14).mean().values
    hh14_1d = Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14_1d = Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = hh14_1d - ll14_1d
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop_14_1d = 100 * np.log10(atr14_1d * np.log10(14) / chop_denom) / np.log10(14)
    
    # Align all indicators to primary timeframe (4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian20 + vol MA20 + chop14 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        donchian_mid = donchian_mid_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        chop_val = chop_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian midline OR chop < 38.2 (trend regime)
            if (price < donchian_mid) or (chop_val < 38.2):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian midline OR chop < 38.2 (trend regime)
            if (price > donchian_mid) or (chop_val < 38.2):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > Donchian high AND volume > 1.5x average AND chop > 61.8 (range)
            if (price > donchian_high) and (vol_ratio_val > 1.5) and (chop_val > 61.8):
                signals[i] = 0.30
                position = 1
            
            # SHORT: Price < Donchian low AND volume > 1.5x average AND chop > 61.8 (range)
            elif (price < donchian_low) and (vol_ratio_val > 1.5) and (chop_val > 61.8):
                signals[i] = -0.30
                position = -1
        
        else:
            signals[i] = position * 0.30  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeConfirmation_1dChopRange_V1"
timeframe = "4h"
leverage = 1.0