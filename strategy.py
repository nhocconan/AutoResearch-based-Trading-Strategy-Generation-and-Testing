#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# Uses 12h Donchian channel breakouts for structure, confirmed by 1d volume spike and low chop (trending regime)
# Designed for low frequency (50-150 trades over 4 years) with clear trend-following logic
# Works in bull markets via breakouts and in bear markets via breakdowns with volume confirmation

name = "12h_Donchian20_1dVolume_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Volume average (20-period) for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d Chopiness Index (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14) - smoothed TR
    atr_14 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        elif i == 14:
            atr_14[i] = np.nanmean(tr[1:15])  # First ATR: average of first 14 TR values
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Sum of ATR over 14 periods
    sum_atr_14 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            sum_atr_14[i] = np.nan
        else:
            sum_atr_14[i] = np.nansum(atr_14[i-13:i+1])
    
    # Chopiness Index: log10(sum(ATR)/log10(14)) / log10(highest_high - lowest_low over 14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop_raw = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14 or sum_atr_14[i] <= 0 or chop_denom[i] <= 0:
            chop_raw[i] = np.nan
        else:
            chop_raw[i] = 100 * np.log10(sum_atr_14[i] / chop_denom[i]) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # 12h Donchian Channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(20, 20)  # Need Donchian and 1d indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 1d average volume
        # Approximate 1d volume by using the aligned 1d MA (conservative)
        vol_confirm = volume[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # Chop regime filter: chop < 50 indicates trending regime (favor breakouts)
        chop_filter = chop_aligned[i] < 50
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price above upper Donchian band with volume and chop confirmation
            if close[i] > highest_high_20[i] and vol_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price below lower Donchian band with volume and chop confirmation
            elif close[i] < lowest_low_20[i] and vol_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when price reaches middle of Donchian channel or opposite signal
            middle_band = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] <= middle_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price reaches middle of Donchian channel or opposite signal
            middle_band = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] >= middle_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals