#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian upper band AND volume > 1.5x average volume AND choppiness > 61.8 (range).
# Short when price breaks below Donchian lower band AND volume > 1.5x average volume AND choppiness > 61.8 (range).
# Exit when price returns to Donchian middle band (mean reversion in ranging markets).
# Uses discrete position size 0.25. Choppiness filter ensures we only trade breakouts in ranging markets
# where mean reversion is effective, avoiding strong trends where breakouts fail.
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets (catch upside breakouts in ranges) and bear markets (catch downside breakouts in ranges).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period) ===
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # === Volume Confirmation ===
    # Volume > 1.5x 20-period average volume
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / volume_ma  # Current volume / average volume
    
    # === Choppiness Index Regime Filter (14-period) ===
    # Measures whether market is ranging (high values) or trending (low values)
    # CHOP > 61.8 = ranging market (good for mean reversion breakout fade)
    # CHOP < 38.2 = trending market (avoid breakouts here)
    # True Range = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR-like calculation: sum of true range over 14 periods
    tr_series = pd.Series(true_range)
    atr14 = tr_series.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = high_series.rolling(window=14, min_periods=14).max().values
    ll14 = low_series.rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    # Avoid division by zero when HH14 == LL14
    range_14 = hh14 - ll14
    choppiness = np.full_like(close, 50.0)  # Default to neutral
    mask = (range_14 > 0) & (~np.isnan(range_14)) & (~np.isnan(atr14))
    choppiness[mask] = 100 * np.log10(atr14[mask] / range_14[mask]) / np.log10(14)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian20 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(choppiness[i]) or volume_ma[i] == 0):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_ratio = volume_ratio[i]
        chop = choppiness[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to middle band (mean reversion)
            if price <= donchian_middle[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to middle band (mean reversion)
            if price >= donchian_middle[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper band AND volume confirmation AND ranging market
            if (price > donchian_upper[i]) and (vol_ratio > 1.5) and (chop > 61.8):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower band AND volume confirmation AND ranging market
            elif (price < donchian_lower[i]) and (vol_ratio > 1.5) and (chop > 61.8):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeConfirmation_ChoppinessFilter_V1"
timeframe = "4h"
leverage = 1.0