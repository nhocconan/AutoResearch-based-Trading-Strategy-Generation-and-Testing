#!/usr/bin/env python3
# 6H_TRIX_VolumeSpike_TrendFilter
# Hypothesis: TRIX (triple exponential average) momentum on 6h with 12h trend filter and volume spike confirmation.
# TRIX filters noise and identifies momentum shifts; 12h trend ensures alignment with higher timeframe direction.
# Volume spike confirms institutional participation. Designed for low trade frequency (15-30/year) to minimize fee drag.
# Works in bull/bear by following 12h trend direction only.

name = "6H_TRIX_VolumeSpike_TrendFilter"
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
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h close for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate TRIX (15,9,9) on 6h close
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9) - 1, then % change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_raw = ema3
    # Calculate % change: (current - previous) / previous * 100
    trix = np.zeros_like(trix_raw)
    trix[1:] = (trix_raw[1:] - trix_raw[:-1]) / trix_raw[:-1] * 100
    # First value remains 0 (no previous)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(15+9+9, 20, 34)  # TRIX needs 33 bars, vol MA needs 20, EMA34 needs 34
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike filter
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: TRIX turns up (>0) + 12h uptrend (price > EMA34) + volume spike
            if (trix[i] > 0 and 
                close[i] > ema_34_12h_aligned[i] and   # 12h uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: TRIX turns down (<0) + 12h downtrend (price < EMA34) + volume spike
            elif (trix[i] < 0 and 
                  close[i] < ema_34_12h_aligned[i] and   # 12h downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: TRIX crosses zero (momentum shift)
            if (position == 1 and trix[i] < 0) or (position == -1 and trix[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals