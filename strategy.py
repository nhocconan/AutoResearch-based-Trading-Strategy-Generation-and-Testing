#!/usr/bin/env python3
# 4h_1d_trix_volume_regime_v1
# Strategy: 4h TRIX with volume confirmation and 1d Choppiness regime filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: TRIX filters noise and identifies momentum. In trending regimes (CHOP < 38.2), go long when TRIX crosses above zero with volume confirmation, short when crosses below zero. In ranging regimes (CHOP > 61.8), fade extreme TRIX readings (>0.1 or <-0.1) with volume confirmation. Uses 1d Choppiness to avoid whipsaws. Low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed DM and TR
    def smooth_series(series, period):
        smoothed = np.zeros_like(series)
        smoothed[period-1] = np.nansum(series[:period])
        for i in range(period, len(series)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + series[i]
        return smoothed
    
    tr_smoothed = smooth_series(tr, 14)
    dm_plus_smoothed = smooth_series(dm_plus, 14)
    dm_minus_smoothed = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and Choppiness
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    chop = 100 * np.log10(tr_smoothed.sum() / (np.abs(di_plus - di_minus).sum() + 1e-10)) / np.log10(14)
    # Simplified: use standard chop calculation
    chop = 100 * np.log10(atr.sum() / (np.abs(di_plus - di_minus).sum() + 1e-10)) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Align Choppiness to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h TRIX (15-period)
    # EMA1
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 100 * (EMA3 - prev_EMA3) / prev_EMA3
    trix = np.zeros_like(close)
    trix[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix[0] = 0
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        chop_val = chop_aligned[i]
        trix_val = trix[i]
        
        # Regime-based logic
        if chop_val < 38.2:  # Trending regime
            # Long: TRIX crosses above zero with volume
            if i > 0 and trix_val > 0 and trix[i-1] <= 0 and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short: TRIX crosses below zero with volume
            elif i > 0 and trix_val < 0 and trix[i-1] >= 0 and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit: TRIX returns to zero
            elif position == 1 and trix_val <= 0:
                position = 0
                signals[i] = 0.0
            elif position == -1 and trix_val >= 0:
                position = 0
                signals[i] = 0.0
        elif chop_val > 61.8:  # Ranging regime
            # Long: TRIX < -0.1 (oversold) with volume
            if trix_val < -0.1 and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            # Short: TRIX > 0.1 (overbought) with volume
            elif trix_val > 0.1 and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit: TRIX returns to neutral zone
            elif position == 1 and trix_val >= -0.05:
                position = 0
                signals[i] = 0.0
            elif position == -1 and trix_val <= 0.05:
                position = 0
                signals[i] = 0.0
        else:  # Transition regime - hold or flat
            signals[i] = 0.0
            position = 0
        # Hold position if no action taken
        if chop_val < 38.2 and position == 1 and trix_val > 0:
            signals[i] = 0.25
        elif chop_val < 38.2 and position == -1 and trix_val < 0:
            signals[i] = -0.25
        elif chop_val > 61.8 and ((position == 1 and trix_val < -0.05) or (position == -1 and trix_val > 0.05)):
            signals[i] = 0.25 if position == 1 else -0.25
    
    return signals