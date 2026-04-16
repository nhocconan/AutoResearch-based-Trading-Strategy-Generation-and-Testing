#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout + 1d volume spike + choppiness regime filter.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND chop > 61.8 (ranging).
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND chop > 61.8 (ranging).
# Exit when price crosses Donchian midline (10-period average of high/low).
# Uses discrete position size 0.25. Targets mean reversion in ranging markets with volume confirmation.
# 4h timeframe targets 20-50 trades/year to minimize fee drag.
# Works in ranging markets (mean reversion) and avoids trending markets via chop filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume average and Choppiness Index ===
    # Volume 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    # True range for chop calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = tr1[0]
    # Sum of true range over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max and min close over 14 periods
    max_close_14 = pd.Series(close_1d).rolling(window=14, min_periods=14).max().values
    min_close_14 = pd.Series(close_1d).rolling(window=14, min_periods=14).min().values
    # Choppiness Index: 100 * log10(sum(tr14) / (max_close - min_close)) / log10(14)
    # Avoid division by zero
    range_14 = max_close_14 - min_close_14
    chop = np.where(range_14 > 0, 100 * np.log10(atr_14 / range_14) / np.log10(14), 50)
    
    # === Primary timeframe indicators: Donchian channels ===
    # Donchian(20) high and low
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midline (10-period average of high/low)
    donch_mid = (pd.Series(high).rolling(window=10, min_periods=10).mean().values + 
                 pd.Series(low).rolling(window=10, min_periods=10).mean().values) / 2
    
    # Align 1d indicators to primary timeframe (4h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20  # Donchian20 needs 20 periods
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(volume_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_ma = vol_ma_aligned[i]
        vol = volume_aligned[i]
        chop_val = chop_aligned[i]
        donch_high = donch_high_20[i]
        donch_low = donch_low_20[i]
        donch_mid_val = donch_mid[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price crosses below Donchian midline
            if price < donch_mid_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price crosses above Donchian midline
            if price > donch_mid_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_confirm = vol > 1.5 * vol_ma
            # Chop filter: ranging market (chop > 61.8)
            chop_filter = chop_val > 61.8
            
            # LONG: Price breaks above Donchian(20) high + volume confirm + chop filter
            if (price > donch_high) and volume_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian(20) low + volume confirm + chop filter
            elif (price < donch_low) and volume_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeSpike_ChopFilter_MeanReversion_V1"
timeframe = "4h"
leverage = 1.0