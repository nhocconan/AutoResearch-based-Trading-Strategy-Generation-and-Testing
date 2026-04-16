#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Bollinger Band squeeze breakout with volume confirmation.
# Long when price breaks above upper BB(20,2) + volume > 1.5x 20-period median volume + ATR(14) > 0.3 * ATR(50).
# Short when price breaks below lower BB(20,2) + volume > 1.5x 20-period median volume + ATR(14) > 0.3 * ATR(50).
# Uses discrete position size 0.25. Exits when price returns to middle BB(20) or ATR contraction (ATR(14) < 0.2 * ATR(50)).
# Bollinger Band squeeze identifies low volatility breakout setups; volume confirms breakout strength; ATR filter ensures volatility expansion.
# 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by capturing volatility expansion breakouts from squeezed conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Bollinger Bands and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # === 1d Indicators: Bollinger Bands (20,2) ===
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20  # 20-period SMA
    
    # === 1d Indicators: ATR (14-period and 50-period for expansion/contraction filter) ===
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    true_range_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    atr_14_1d = pd.Series(true_range_1d).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(true_range_1d).rolling(window=50, min_periods=50).mean().values
    
    # === 1d Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (6h)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1d, middle_bb)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 50, 20)  # BB20 needs 20, ATR50 needs 50, volume median needs 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or np.isnan(middle_bb_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or
            np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        middle_bb_val = middle_bb_aligned[i]
        atr_14 = atr_14_aligned[i]
        atr_50 = atr_50_aligned[i]
        vol_median = vol_median_aligned[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.5x median volume
        volume_spike = current_vol_1d > (vol_median * 1.5)
        
        # ATR expansion filter: ATR(14) > 0.3 * ATR(50) (volatility expansion)
        atr_expansion = atr_14 > (atr_50 * 0.3)
        
        # ATR contraction filter: ATR(14) < 0.2 * ATR(50) (low volatility)
        atr_contraction = atr_14 < (atr_50 * 0.2)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to middle BB OR ATR contraction
            if (price <= middle_bb_val) or atr_contraction:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to middle BB OR ATR contraction
            if (price >= middle_bb_val) or atr_contraction:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper BB + volume spike + ATR expansion
            if (price > upper_bb_val) and volume_spike and atr_expansion:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower BB + volume spike + ATR expansion
            elif (price < lower_bb_val) and volume_spike and atr_expansion:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dBollingerBandSqueezeBreakout_VolumeSpike1.5x_ATRexpansion0.3x_EXITmiddleBB_ATRcontraction0.2x_v1"
timeframe = "6h"
leverage = 1.0