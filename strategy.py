# 4H_DONCHIAN20_VOLUME_CONFIRMATION_CHOP_REGIME_V1
# Hypothesis: 4H Donchian(20) breakout with volume confirmation and Choppiness Index regime filter.
# Long when price breaks above 20-period high + volume > 1.5x average + CHOP < 61.8 (trending).
# Short when price breaks below 20-period low + volume > 1.5x average + CHOP < 61.8 (trending).
# Uses daily Choppiness Index to avoid ranging markets where breakouts fail.
# Target: 15-35 trades/year per symbol.
# Works in bull (catch breakouts) and bear (avoid false breakouts in ranges via CHOP filter).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range: max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR(14)
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    sum_atr = np.zeros_like(atr)
    for i in range(13, len(atr)):
        if i == 13:
            sum_atr[i] = np.sum(atr[:14])
        else:
            sum_atr[i] = sum_atr[i-1] - atr[i-14] + atr[i]
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros_like(high_1d)
    min_low = np.zeros_like(low_1d)
    for i in range(len(high_1d)):
        if i < 13:
            max_high[i] = np.nan
            min_low[i] = np.nan
        else:
            max_high[i] = np.max(high_1d[i-13:i+1])
            min_low[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR)/ (max(high)-min(low))) / log10(14)
    range_high_low = max_high - min_low
    chop = np.full_like(close_1d, 50.0, dtype=float)
    for i in range(13, len(chop)):
        if range_high_low[i] > 0 and sum_atr[i] > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / range_high_low[i]) / np.log10(14)
    
    # Align Choppiness Index to 4H timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4H data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(len(high)):
        if i < 19:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 19:
            vol_ma_20[i] = np.nan
        else:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    vol_ratio = np.full_like(volume, 0.0)
    for i in range(len(volume)):
        if not np.isnan(vol_ma_20[i]) and vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
    
    vol_filter = vol_ratio > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        vol_ok = vol_filter[i]
        
        # Regime filter: trending market (CHOP < 61.8)
        trending = chop_val < 61.8
        
        if position == 0:
            # Long: break above Donchian high + volume + trending
            if price > highest_high[i] and vol_ok and trending:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + volume + trending
            elif price < lowest_low[i] and vol_ok and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or chop > 61.8 (range)
            if price < lowest_low[i] or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or chop > 61.8 (range)
            if price > highest_high[i] or chop_val >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_VolumeConfirmation_ChopRegime_V1"
timeframe = "4h"
leverage = 1.0