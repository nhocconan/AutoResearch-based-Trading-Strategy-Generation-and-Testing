#!/usr/bin/env python3
# 12h_TRIX_VolumeSpike_TrendFilter
# Hypothesis: 12-hour TRIX momentum oscillator with daily trend filter and volume spike confirmation.
# TRIX > 0 indicates bullish momentum, TRIX < 0 indicates bearish momentum.
# Daily trend filter (price > daily EMA50) prevents counter-trend trades.
# Volume spike (1.5x average volume) confirms momentum.
# Designed for low trade frequency (<30/year) to minimize fee drag in bear markets.

name = "12h_TRIX_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX (15-period) on 12h data
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - then percentage change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.full(n, np.nan)
    trix[14:] = (ema3[14:] - ema3[13:-1]) / ema3[13:-1] * 100  # Percentage change
    
    # Volume filter: average volume (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 15*3)  # Ensure we have TRIX and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (1.5x average volume)
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: TRIX crosses above zero + daily uptrend + volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and  # Bullish crossover
                close[i] > ema_50_1d_aligned[i] and   # Daily uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + daily downtrend + volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and  # Bearish crossover
                  close[i] < ema_50_1d_aligned[i] and   # Daily downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: TRIX returns to zero (momentum fade)
            if (position == 1 and trix[i] < 0) or (position == -1 and trix[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals