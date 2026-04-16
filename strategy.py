#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above 1d Donchian upper (20) AND volume > 1.5x MA20 volume AND chop < 61.8 (trending).
# Short when price breaks below 1d Donchian lower (20) AND volume > 1.5x MA20 volume AND chop < 61.8 (trending).
# Uses discrete position size 0.25. Donchian provides structure, volume confirms conviction, chop filter avoids whipsaws in ranging markets.
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets (capture breakouts) and bear markets (capture breakdowns) with volume confirmation reducing false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Donchian Channels (20) ===
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: EMA20 for volume filter ===
    volume_1d = df_1d['volume'].values
    volume_ema20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 1d Indicators: Choppiness Index (14) ===
    # CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0]  # first bar
    tr3[0] = tr2[0]  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14 * 14 / (max_high14 - min_low14 + 1e-10)) / np.log10(14)
    choppiness = chop_raw  # already in 0-100 range
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    volume_ema20_aligned = align_htf_to_ltf(prices, df_1d, volume_ema20)
    choppiness_aligned = align_htf_to_ltf(prices, df_1d, choppiness)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian20 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_ema20_aligned[i]) or np.isnan(choppiness_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_ema = volume_ema20_aligned[i]
        chop = choppiness_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian lower (breakdown) OR chop > 61.8 (ranging)
            if (price < lower) or (chop > 61.8):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian upper (breakout) OR chop > 61.8 (ranging)
            if (price > upper) or (chop > 61.8):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5 x EMA20 volume
            volume_confirm = vol > 1.5 * vol_ema
            
            # LONG: Price breaks above Donchian upper AND volume confirm AND chop < 61.8 (trending)
            if (price > upper) and volume_confirm and (chop < 61.8):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND volume confirm AND chop < 61.8 (trending)
            elif (price < lower) and volume_confirm and (chop < 61.8):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dDonchian20_VolumeConfirmation_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0