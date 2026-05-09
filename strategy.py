#!/usr/bin/env python3
# 4h_Trix_VolumeSpike_Regime
# Hypothesis: TRIX (12-period) crossing above/below zero with volume >2x 20-bar average and chop regime filter (Choppiness Index > 61.8 for mean reversion, < 38.2 for trend) to capture momentum in trending markets and reversals in ranging markets. Works in both bull and bear markets by adapting to regime.

name = "4h_Trix_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index (needs daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) for Choppiness Index
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[0:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR14 over 14 periods) / (max(high) - min(low) over 14 periods)) / log10(14)
    sum_atr_14 = np.full_like(atr_14, np.nan)
    max_high_14 = np.full_like(high_1d, np.nan)
    min_low_14 = np.full_like(low_1d, np.nan)
    
    if len(atr_14) >= 14:
        for i in range(14, len(atr_14)):
            sum_atr_14[i] = np.sum(atr_14[i-13:i+1])
    
    if len(high_1d) >= 14:
        for i in range(14, len(high_1d)):
            max_high_14[i] = np.max(high_1d[i-13:i+1])
            min_low_14[i] = np.min(low_1d[i-13:i+1])
    
    chop = np.full_like(close_1d, np.nan)
    valid_chop = (~np.isnan(sum_atr_14)) & (~np.isnan(max_high_14)) & (~np.isnan(min_low_14)) & ((max_high_14 - min_low_14) != 0)
    chop[valid_chop] = 100 * np.log10(sum_atr_14[valid_chop] / (max_high_14[valid_chop] - min_low_14[valid_chop])) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe (needs 2-bar extra delay for confirmation)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=2)
    
    # Get 1d data for TRIX calculation (using close)
    # TRIX: EMA(EMA(EMA(close, 12), 12), 12) then % change
    def ema(array, period):
        result = np.full_like(array, np.nan)
        if len(array) >= period:
            multiplier = 2 / (period + 1)
            result[period-1] = np.mean(array[0:period])
            for i in range(period, len(array)):
                result[i] = (array[i] * multiplier) + (result[i-1] * (1 - multiplier))
        return result
    
    close_1d_for_trix = close_1d
    ema1 = ema(close_1d_for_trix, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    trix = np.full_like(close_1d, np.nan)
    valid_trix = ~np.isnan(ema3)
    trix[valid_trix] = (ema3[valid_trix] - np.roll(ema3, 1)[valid_trix]) / np.roll(ema3, 1)[valid_trix] * 100
    trix[0] = np.nan  # First value has no previous
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume filter: 4h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND volume confirmation AND trending regime (CHOP < 38.2)
            if i > 0 and trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and volume_ratio[i] > 2.0 and chop_aligned[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND volume confirmation AND trending regime (CHOP < 38.2)
            elif i > 0 and trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and volume_ratio[i] > 2.0 and chop_aligned[i] < 38.2:
                signals[i] = -0.25
                position = -1
            # Mean reversion in ranging market: TRIX extreme + volume spike + CHOP > 61.8
            elif volume_ratio[i] > 2.0 and chop_aligned[i] > 61.8:
                if trix_aligned[i] < -0.5:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif trix_aligned[i] > 0.5:  # Overbought
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR chop regime shifts to ranging (CHOP > 61.8) for mean reversion exit
            if trix_aligned[i] < 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR chop regime shifts to ranging (CHOP > 61.8) for mean reversion exit
            if trix_aligned[i] > 0 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals