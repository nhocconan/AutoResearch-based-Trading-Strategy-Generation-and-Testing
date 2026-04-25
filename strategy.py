#!/usr/bin/env python3
"""
4h TRIX + Volume Spike + Choppiness Regime Filter
Hypothesis: TRIX (Triple Exponential Average) filters noise and shows smoothed momentum.
Long when TRIX crosses above zero with volume spike in non-choppy market (CHOP > 61.8 = ranging, < 38.2 = trending).
Short when TRIX crosses below zero with volume spike in trending market.
Volume spike confirms institutional participation. Works in both bull and bear markets by following momentum.
4h timeframe targets 19-50 trades/year (75-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate TRIX(12,9,9) - triple smoothed EMA of ROC
    # TRIX = EMA(EMA(EMA(ROC, 12), 9), 9)
    roc = pd.Series(close).pct_change(periods=1)  # 1-period ROC
    ema1 = roc.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = ema3.values * 100  # scale for readability
    
    # Calculate TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    
    # Regime filters
    chop_choppy = chop > 61.8   # ranging market (mean revert)
    chop_trending = chop < 38.2  # trending market (trend follow)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 14, 12+9+9)  # volume MA, CHOP, TRIX
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_trix = trix[i]
        curr_trix_signal = trix_signal[i]
        prev_trix = trix[i-1]
        prev_trix_signal = trix_signal[i-1]
        vol_spike = volume_spike[i]
        curr_chop = chop[i]
        
        # TRIX cross signals
        trix_cross_up = (prev_trix <= prev_trix_signal) and (curr_trix > curr_trix_signal)
        trix_cross_down = (prev_trix >= prev_trix_signal) and (curr_trix < curr_trix_signal)
        
        if position == 0:
            # Look for entry signals
            # Long: TRIX crosses above signal AND volume spike AND trending market (not choppy)
            long_entry = trix_cross_up and vol_spike and chop_trending
            # Short: TRIX crosses below signal AND volume spike AND trending market (not choppy)
            short_entry = trix_cross_down and vol_spike and chop_trending
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: TRIX crosses below signal OR market becomes choppy (range)
            if trix_cross_down or chop_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: TRIX crosses above signal OR market becomes choppy (range)
            if trix_cross_up or chop_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_ChoppinessRegime"
timeframe = "4h"
leverage = 1.0