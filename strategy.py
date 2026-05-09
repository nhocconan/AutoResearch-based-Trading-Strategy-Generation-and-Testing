#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1d_Regime_Filter
Hypothesis: KAMA (adaptive moving average) on 4h defines trend direction, filtered by 1d Choppiness Index (range vs trend).
In trending markets (CHOP < 38.2): follow KAMA direction. In ranging markets (CHOP > 61.8): fade moves at Bollinger Bands.
This avoids whipsaws in sideways markets while capturing trends. Designed for low trade frequency (~20-40/year) to minimize fee drag.
Works in both bull and bear markets via regime adaptation.
"""

name = "4h_KAMA_Trend_With_1d_Regime_Filter"
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
    
    # --- 4h Indicators: KAMA (trend) and Bollinger Bands (mean reversion) ---
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    
    # Recompute volatility properly
    volatility = np.zeros_like(close)
    for i in range(er_len, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_len+1:i+1])))
    
    er = np.zeros_like(close)
    er[er_len:] = change[er_len:] / np.where(volatility[er_len:] == 0, 1, volatility[er_len:])
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]  # seed
    for i in range(er_len + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Bollinger Bands (20, 2)
    bb_len = 20
    bb_mult = 2
    sma = np.full_like(close, np.nan)
    bb_std = np.full_like(close, np.nan)
    upper = np.full_like(close, np.nan)
    lower = np.full_like(close, np.nan)
    
    if len(close) >= bb_len:
        sma[bb_len-1] = np.mean(close[0:bb_len])
        for i in range(bb_len, len(close)):
            sma[i] = sma[i-1] + (close[i] - close[i-bb_len]) / bb_len
        
        for i in range(bb_len-1, len(close)):
            bb_std[i] = np.std(close[i-bb_len+1:i+1])
        
        upper = sma + bb_mult * bb_std
        lower = sma - bb_mult * bb_std
    
    # --- 1d Indicators: Choppiness Index (regime filter) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr_len = 14
    atr = np.full_like(close_1d, np.nan)
    if len(tr) >= atr_len + 1:
        atr[atr_len] = np.nanmean(tr[1:atr_len+1])  # seed
        for i in range(atr_len + 1, len(close_1d)):
            atr[i] = (atr[i-1] * (atr_len - 1) + tr[i]) / atr_len
    
    # Sum of ATR over CHOP period
    chop_len = 14
    atr_sum = np.full_like(close_1d, np.nan)
    if len(atr) >= chop_len:
        for i in range(chop_len-1, len(close_1d)):
            atr_sum[i] = np.nansum(atr[i-chop_len+1:i+1])
    
    # Highest high and lowest low over CHOP period
    highest_high = np.full_like(close_1d, np.nan)
    lowest_low = np.full_like(close_1d, np.nan)
    if len(high_1d) >= chop_len:
        for i in range(chop_len-1, len(high_1d)):
            highest_high[i] = np.max(high_1d[i-chop_len+1:i+1])
            lowest_low[i] = np.min(low_1d[i-chop_len+1:i+1])
    
    # Choppiness Index
    chop = np.full_like(close_1d, 50.0)  # default neutral
    valid = (~np.isnan(atr_sum)) & (atr_sum != 0) & (~np.isnan(highest_high)) & (~np.isnan(lowest_low))
    chop[valid] = 100 * np.log10(atr_sum[valid] / (highest_high[valid] - lowest_low[valid])) / np.log10(chop_len)
    
    # Align 1d indicators to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama[:len(close_1d)])  # dummy align - will replace
    
    # Actually align 4h indicators properly
    # Re-get 4h data for proper alignment (but we already have close)
    # Instead, we'll compute KAMA on 4h directly and it's already aligned
    # Chop is the only true HTF indicator needing alignment
    
    # Volume filter: avoid low liquidity periods
    vol_ma_len = 20
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= vol_ma_len:
        vol_ma[vol_ma_len-1] = np.mean(volume[0:vol_ma_len])
        for i in range(vol_ma_len, len(volume)):
            vol_ma[i] = vol_ma[i-1] + (volume[i] - volume[i-vol_ma_len]) / vol_ma_len
    
    volume_filter = np.ones_like(volume, dtype=bool)
    if len(volume) >= vol_ma_len:
        volume_filter = volume > 0.5 * vol_ma  # at least half average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_len, bb_len, vol_ma_len) + 5
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entry
            if chop_aligned[i] < 38.2:  # Trending market
                # Follow KAMA direction
                if close[i] > kama[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i]:
                    signals[i] = -0.25
                    position = -1
            elif chop_aligned[i] > 61.8:  # Ranging market
                # Mean reversion at Bollinger Bands
                if close[i] <= lower[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= upper[i]:
                    signals[i] = -0.25
                    position = -1
            # Else: neutral chop (38.2-61.8) - stay flat
        
        elif position == 1:  # Long
            # Exit conditions
            if chop_aligned[i] < 38.2:  # Still trending
                if close[i] <= kama[i]:  # Trend reversal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif chop_aligned[i] > 61.8:  # Ranging - take profit at mean
                if close[i] >= sma[i]:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Neutral chop
                if close[i] <= lower[i] or close[i] >= upper[i]:  # Outside bands
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:  # Short
            # Exit conditions
            if chop_aligned[i] < 38.2:  # Still trending
                if close[i] >= kama[i]:  # Trend reversal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif chop_aligned[i] > 61.8:  # Ranging - take profit at mean
                if close[i] <= sma[i]:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Neutral chop
                if close[i] <= lower[i] or close[i] >= upper[i]:  # Outside bands
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals