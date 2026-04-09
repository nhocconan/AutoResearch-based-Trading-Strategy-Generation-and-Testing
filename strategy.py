#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR-based volume confirmation + chop regime filter
# Donchian breakout captures strong momentum moves in both bull and bear markets
# 1d ATR-normalized volume > 1.5x 20-day average filters low-quality breakouts
# Choppiness Index regime: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (mean revert)
# Discrete sizing 0.25 to limit fee drag. Target: 20-50 trades/year per symbol.

name = "4h_1d_donchian_atrvol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ATR(14) for volume normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    atr_1d = np.concatenate([[np.nan], atr_1d])  # align with original indices
    
    # 1d ATR-normalized volume and its 20-day average
    vol_atr_ratio_1d = np.where(atr_1d != 0, volume_1d / atr_1d, 0)
    vol_atr_s_1d = pd.Series(vol_atr_ratio_1d)
    avg_vol_atr_1d = vol_atr_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # 1d Choppiness Index (CHOP)
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_1d[14:]).rolling(window=14, min_periods=14).sum().values
    sum_atr_14 = np.concatenate([np.full(27, np.nan), sum_atr_14])  # align indices
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0,
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14),
                       50)
    
    # Align 1d indicators to 4h
    avg_vol_atr_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_atr_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(avg_vol_atr_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h ATR-normalized volume > 1.5x 1d average
        atr_4h = np.abs(high[i] - low[i])  # approximate 4h ATR
        vol_atr_4h = np.where(atr_4h != 0, volume[i] / atr_4h, 0)
        volume_confirmed = vol_atr_4h > 1.5 * avg_vol_atr_1d_aligned[i]
        
        # Regime filters
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long
            if close[i] < lowest_low[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short
            if close[i] > highest_high[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                if close[i] > highest_high[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_low[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                if close[i] < lowest_low[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > highest_high[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals