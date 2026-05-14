#!/usr/bin/env python3
# 4h_Advanced_Camarilla_Pivot_Squeeze
# Hypothesis: Combines Camarilla pivot levels with Bollinger Band squeeze detection and volume confirmation.
# Uses 1d Camarilla R1/S1 for entry, Bollinger Band width percentile for regime filter (low volatility = squeeze),
# and volume spike for confirmation. Designed to work in both trending and ranging markets by capturing
# breakouts from low volatility phases. Target: 20-35 trades/year per symbol with disciplined risk.

name = "4h_Advanced_Camarilla_Pivot_Squeeze"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Bollinger Bands (20, 2) for squeeze detection
    sma_20 = np.full_like(close_1d, np.nan)
    std_20 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 20:
        sma_20[19] = np.mean(close_1d[0:20])
        std_20[19] = np.std(close_1d[0:20])
        for i in range(20, len(close_1d)):
            sma_20[i] = (sma_20[i-1] * 19 + close_1d[i]) / 20
            std_20[i] = np.sqrt((std_20[i-1]**2 * 19 + (close_1d[i] - sma_20[i])**2) / 20)
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Calculate Bollinger Band width percentile (20-period lookback)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    if len(bb_width) >= 40:
        for i in range(39, len(bb_width)):
            window = bb_width[i-19:i+1]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) >= 10:
                current_val = bb_width[i]
                if not np.isnan(current_val):
                    percentile = np.sum(valid_window <= current_val) / len(valid_window) * 100
                    bb_width_percentile[i] = percentile
    
    # Calculate daily Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    daily_range = high_1d - low_1d
    camarilla_R1 = close_1d + daily_range * 1.1 / 12
    camarilla_S1 = close_1d - daily_range * 1.1 / 12
    
    # Align daily indicators to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
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
        if np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or \
           np.isnan(bb_width_percentile_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: low volatility (BB width in lowest 30% percentile)
        squeeze_condition = bb_width_percentile_aligned[i] < 30
        
        if position == 0:
            # Enter long: Price breaks above Camarilla R1 AND volume confirmation AND volatility squeeze
            if close[i] > camarilla_R1_aligned[i] and volume_ratio[i] > 2.0 and squeeze_condition:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Camarilla S1 AND volume confirmation AND volatility squeeze
            elif close[i] < camarilla_S1_aligned[i] and volume_ratio[i] > 2.0 and squeeze_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: Price breaks below Camarilla S1 OR volatility expansion (end of squeeze)
            if close[i] < camarilla_S1_aligned[i] or bb_width_percentile_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: Price breaks above Camarilla R1 OR volatility expansion (end of squeeze)
            if close[i] > camarilla_R1_aligned[i] or bb_width_percentile_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals