#!/usr/bin/env python3
"""
6h_TRIX_VolumeSpike_Regime
Hypothesis: TRIX (12-period) crosses zero with volume spike confirmation (>2x 24-period average) and regime filter using 12h Choppiness Index (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend following). Works in both bull and bear markets by adapting to market regime. Uses 6h timeframe for execution with 12h regime filter to reduce whipsaws.
"""

name = "6h_TRIX_VolumeSpike_Regime"
timeframe = "6h"
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
    
    # Get 12h data for regime filter (Choppiness Index)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate TRIX (12-period) on close
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        multiplier = 2 / (period + 1)
        result[period-1] = np.mean(arr[0:period])
        for i in range(period, len(arr)):
            result[i] = (arr[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    ema1 = ema(close, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    
    # TRIX = (EMA3[i] - EMA3[i-1]) / EMA3[i-1] * 100
    trix = np.full_like(close, np.nan)
    valid = (~np.isnan(ema3)) & (~np.roll(np.isnan(ema3), 1)) & (np.roll(ema3, 1) != 0)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix[0] = np.nan
    
    # Calculate 12h Choppiness Index (CHOP) for regime filter
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Shift close to get previous close
    prev_close_12h = np.roll(close_12h, 1)
    prev_close_12h[0] = close_12h[0]  # first value
    
    tr = true_range(high_12h, low_12h, prev_close_12h)
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[0:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate highest high and lowest low over 14 periods
    hh_14 = np.full_like(high_12h, np.nan)
    ll_14 = np.full_like(low_12h, np.nan)
    if len(high_12h) >= 14:
        for i in range(14, len(high_12h)):
            hh_14[i] = np.max(high_12h[i-13:i+1])
            ll_14[i] = np.min(low_12h[i-13:i+1])
    
    # CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    chop = np.full_like(close_12h, np.nan)
    valid = (~np.isnan(atr_14)) & (~np.isnan(hh_14)) & (~np.isnan(ll_14)) & ((hh_14 - ll_14) != 0)
    if np.any(valid):
        sum_atr = np.nansum(atr_14)  # This needs to be rolling sum
        # Recalculate with rolling sum
        sum_atr_14 = np.full_like(tr, np.nan)
        if len(tr) >= 14:
            sum_atr_14[13] = np.sum(tr[0:14])
            for i in range(14, len(tr)):
                sum_atr_14[i] = sum_atr_14[i-1] - tr[i-14] + tr[i]
        
        valid_chop = (~np.isnan(sum_atr_14)) & (~np.isnan(hh_14)) & (~np.isnan(ll_14)) & ((hh_14 - ll_14) != 0)
        chop[valid_chop] = 100 * np.log10(sum_atr_14[valid_chop] / (hh_14[valid_chop] - ll_14[valid_chop])) / np.log10(14)
    
    # Align 12h CHOP to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume spike filter: current volume / 24-period average volume (24*6h = 6 days)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(24, 34)  # Ensure volume MA and TRIX are ready (TRIX needs ~34 bars)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Regime-based entry logic
            if chop_aligned[i] > 61.8:  # Ranging market - mean reversion
                # Long: TRIX crosses above zero AND volume spike
                if trix[i] > 0 and trix[i-1] <= 0 and volume_ratio[i] > 2.0:
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                # Short: TRIX crosses below zero AND volume spike
                elif trix[i] < 0 and trix[i-1] >= 0 and volume_ratio[i] > 2.0:
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
            elif chop_aligned[i] < 38.2:  # Trending market - trend following
                # Long: TRIX > 0 AND rising AND volume spike
                if trix[i] > 0 and trix[i] > trix[i-1] and volume_ratio[i] > 2.0:
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                # Short: TRIX < 0 AND falling AND volume spike
                elif trix[i] < 0 and trix[i] < trix[i-1] and volume_ratio[i] > 2.0:
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit conditions
                if chop_aligned[i] > 61.8:  # In range, exit on TRIX cross below zero
                    if trix[i] < 0 and trix[i-1] >= 0:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    else:
                        signals[i] = 0.25
                else:  # In trend, exit on TRIX deterioration
                    if trix[i] <= 0 or trix[i] < trix[i-1]:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    else:
                        signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit conditions
                if chop_aligned[i] > 61.8:  # In range, exit on TRIX cross above zero
                    if trix[i] > 0 and trix[i-1] <= 0:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    else:
                        signals[i] = -0.25
                else:  # In trend, exit on TRIX deterioration
                    if trix[i] >= 0 or trix[i] > trix[i-1]:
                        signals[i] = 0.0
                        position = 0
                        bars_since_entry = 0
                    else:
                        signals[i] = -0.25
    
    return signals