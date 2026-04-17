#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_Regime
Strategy: TRIX momentum + volume spike + chop regime filter.
Long: TRIX crosses above zero + volume > 2x average + CHOP > 61.8 (range)
Short: TRIX crosses below zero + volume > 2x average + CHOP > 61.8 (range)
Exit: TRIX crosses zero in opposite direction or CHOP < 38.2 (trend)
Position size: 0.25
Designed to capture mean reversion in ranging markets with momentum confirmation.
Timeframe: 4h
"""

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
    
    # Calculate TRIX (15-period)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1 period ago
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change(1) * 100  # percentage change
    trix_values = trix.values
    trix_prev = np.roll(trix_values, 1)
    trix_prev[0] = np.nan
    
    # Calculate CHOPPINESS INDEX (14-period)
    # CHOP = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    max_low = max_high - min_low
    
    # Avoid division by zero
    chop_raw = np.divide(
        pd.Series(atr).rolling(window=14, min_periods=14).sum().values,
        max_low,
        out=np.full_like(max_low, np.nan),
        where=max_low!=0
    )
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Volume confirmation (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # TRIX needs ~30, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_values[i]) or 
            np.isnan(trix_prev[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Chop regime filter: CHOP > 61.8 (range-bound market)
        chop_filter = chop[i] > 61.8
        
        # TRIX crossover signals
        trix_cross_up = trix_values[i] > 0 and trix_prev[i] <= 0
        trix_cross_down = trix_values[i] < 0 and trix_prev[i] >= 0
        
        # Exit conditions: TRIX cross opposite or chop < 38.2 (trending)
        exit_long = trix_cross_down or chop[i] < 38.2
        exit_short = trix_cross_up or chop[i] < 38.2
        
        if position == 0:
            # Long: TRIX crosses up + volume + chop (range)
            if trix_cross_up and volume_filter and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses down + volume + chop (range)
            elif trix_cross_down and volume_filter and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX cross down or chop < 38.2 (trend)
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX cross up or chop < 38.2 (trend)
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0