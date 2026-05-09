#/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_Regime
Hypothesis: TRIX(12) momentum + volume spike (>2x 24-period average) + Choppiness regime filter (CHOP<38.2 = trending) on 12h timeframe.
TRIX filters noise and captures sustained momentum. Volume confirms breakout strength.
Choppiness regime ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe.
Works in both bull and bear markets by following momentum in trending regimes.
"""

name = "12h_TRIX_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX and Choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate TRIX(12) - triple smoothed EMA of ROC
    # TRIX = EMA(EMA(EMA(ROC), 12), 12), 12)
    roc = np.full_like(close_1d, np.nan)
    roc[12:] = (close_1d[12:] - close_1d[:-12]) / close_1d[:-12] * 100
    
    # Triple EMA smoothing
    ema1 = np.full_like(close_1d, np.nan)
    ema2 = np.full_like(close_1d, np.nan)
    ema3 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 12:
        # First EMA
        ema1[11] = np.mean(roc[0:12])
        for i in range(12, len(roc)):
            if not np.isnan(roc[i]):
                ema1[i] = (roc[i] * 2 + ema1[i-1] * 10) / 12  # alpha = 2/(12+1)
        
        # Second EMA
        ema2[22] = np.mean(ema1[12:23])
        for i in range(23, len(ema1)):
            if not np.isnan(ema1[i]):
                ema2[i] = (ema1[i] * 2 + ema2[i-1] * 10) / 12
        
        # Third EMA (TRIX)
        ema3[33] = np.mean(ema2[23:34])
        for i in range(34, len(ema2)):
            if not np.isnan(ema2[i]):
                ema3[i] = (ema2[i] * 2 + ema3[i-1] * 10) / 12
    
    trix = ema3  # TRIX values
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Calculate Choppiness Index(14) for regime detection
    # CHOP = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    atr = np.full_like(close_1d, np.nan)
    tr = np.maximum(np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])), np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:14])  # First ATR
        for i in range(14, len(tr)):
            if not np.isnan(tr[i]):
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate rolling max(high) and min(low) for 14 periods
    max_high = np.full_like(high_1d, np.nan)
    min_low = np.full_like(low_1d, np.nan)
    
    for i in range(13, len(high_1d)):
        max_high[i] = np.max(high_1d[i-13:i+1])
        min_low[i] = np.min(low_1d[i-13:i+1])
    
    chop = np.full_like(close_1d, 50.0)  # Default neutral
    valid = (~np.isnan(atr)) & (~np.isnan(max_high)) & (~np.isnan(min_low)) & ((max_high - min_low) > 0)
    chop[valid] = 100 * np.log10(nansum(atr[np.maximum(0, i-13):i+1]) / (max_high[i] - min_low[i])) / np.log10(14)
    
    # Align Choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike filter: current volume / 24-period average volume (24*12h = 12 days)
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
    
    start_idx = max(34, 24)  # Ensure TRIX and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: TRIX > 0 (bullish momentum) AND volume spike AND trending regime (CHOP < 38.2)
            if (trix_aligned[i] > 0 and 
                volume_ratio[i] > 2.0 and 
                chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: TRIX < 0 (bearish momentum) AND volume spike AND trending regime (CHOP < 38.2)
            elif (trix_aligned[i] < 0 and 
                  volume_ratio[i] > 2.0 and 
                  chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: TRIX turns negative OR chop becomes ranging (CHOP > 61.8)
                if trix_aligned[i] < 0 or chop_aligned[i] > 61.8:
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
                # Exit short: TRIX turns positive OR chop becomes ranging (CHOP > 61.8)
                if trix_aligned[i] > 0 or chop_aligned[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

def nansum(arr):
    """Helper function to sum array ignoring NaNs"""
    if len(arr) == 0:
        return 0
    return np.nansum(arr) if np.any(~np.isnan(arr)) else 0