#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_RegimeAdaptive
Hypothesis: Adaptive Camarilla R3/S3 breakout strategy that uses regime detection to switch between breakout and mean reversion. In trending regimes (CHOP < 38.2), trade breakouts with volume confirmation. In ranging regimes (CHOP > 61.8), trade mean reversion at S3/R3 levels with volume confirmation. Uses 1d HTF for Camarilla levels. Fixed size 0.25 to control trade frequency (~25-35 trades/year). Designed to work in both bull and bear markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (stronger breakout levels)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)   # S3 level
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2  # Midpoint for mean reversion
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Volume confirmation: volume > 60th percentile of 30-period lookback
    vol_series = pd.Series(volume)
    vol_percentile_60 = vol_series.rolling(window=30, min_periods=30).quantile(0.60).values
    volume_confirm = volume > vol_percentile_60
    
    # Choppiness regime: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid log(0) when highest == lowest (add tiny epsilon)
    diff = highest_high_14 - lowest_low_14
    diff = np.where(diff == 0, 1e-10, diff)
    chop = 100 * np.log10(atr_14 * 14 / np.log(diff)) / np.log10(14)
    chop_regime_trending = chop < 38.2   # trending market
    chop_regime_ranging = chop > 61.8    # ranging market
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (30 for volume percentile, 14 for ATR/CHOP)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or 
            np.isnan(vol_percentile_60[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        camarilla_mid_val = camarilla_mid_aligned[i]
        vol_conf = volume_confirm[i]
        chop_trend = chop_regime_trending[i]
        chop_range = chop_regime_ranging[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry based on regime
            if chop_trend and vol_conf:
                # Trending regime: breakout strategy
                long_entry = close_val > camarilla_r3_val
                short_entry = close_val < camarilla_s3_val
                if long_entry:
                    signals[i] = size
                    position = 1
                elif short_entry:
                    signals[i] = -size
                    position = -1
            elif chop_range and vol_conf:
                # Ranging regime: mean reversion at extremes
                long_entry = close_val < camarilla_s3_val  # Buy near support
                short_entry = close_val > camarilla_r3_val  # Sell near resistance
                if long_entry:
                    signals[i] = size
                    position = 1
                elif short_entry:
                    signals[i] = -size
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position
            if chop_trend:
                # In trending regime: exit on breakout failure (re-enter if strong)
                if close_val < camarilla_mid_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:
                # In ranging regime: exit at midpoint (mean reversion target)
                if close_val > camarilla_mid_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Short position
            if chop_trend:
                # In trending regime: exit on breakdown failure
                if close_val > camarilla_mid_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:
                # In ranging regime: exit at midpoint
                if close_val < camarilla_mid_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_RegimeAdaptive"
timeframe = "4h"
leverage = 1.0