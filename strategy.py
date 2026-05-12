#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_Regime
# Hypothesis: TRIX momentum combined with volume spikes and Choppiness regime filter.
# TRIX captures short-term momentum, volume confirms strength, and Choppiness distinguishes trending/ranging markets.
# In trending markets (CHOP < 38.2), follow TRIX crossovers; in ranging (CHOP > 61.8), fade extremes.
# Designed for low frequency (20-40 trades/year) to survive both bull and bear markets by adapting to regime.

name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === TRIX (1-period ROC of EMA smoothed 3x) ===
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change(periods=1))
    trix_signal = trix.ewm(span=8, adjust=False, min_periods=8).mean()
    trix_hist = trix - trix_signal
    
    # === ATR for volatility ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Choppiness Index (14-period) ===
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # === Volume spike (20-period avg) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(trix_hist[i]) or np.isnan(trix_signal[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter
        trending = chop[i] < 38.2  # Trending market
        ranging = chop[i] > 61.8   # Ranging market
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: TRIX histogram crosses above zero in trending market OR oversold bounce in ranging
            if trending and trix_hist[i] > 0 and trix_hist[i-1] <= 0 and vol_ok:
                signals[i] = 0.25
                position = 1
            elif ranging and trix[i] < -0.5 and trix[i] > trix[i-1] and vol_ok:  # Oversold bounce
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX histogram crosses below zero in trending market OR overbought fade in ranging
            elif trending and trix_hist[i] < 0 and trix_hist[i-1] >= 0 and vol_ok:
                signals[i] = -0.25
                position = -1
            elif ranging and trix[i] > 0.5 and trix[i] < trix[i-1] and vol_ok:  # Overbought fade
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: TRIX histogram crosses below zero or volatility expansion
            if trix_hist[i] < 0 and trix_hist[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX histogram crosses above zero or volatility expansion
            if trix_hist[i] > 0 and trix_hist[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals