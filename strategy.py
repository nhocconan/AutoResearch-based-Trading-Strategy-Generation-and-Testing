#!/usr/bin/env python3
"""
4h_Trix_ZeroCross_VolumeSpike_12hTrend
Hypothesis: TRIX zero cross (momentum) with volume confirmation and 12h trend filter.
Works in both bull and bear markets by capturing momentum shifts with volume validation.
Designed for low trade frequency (20-50/year) to minimize fee drag.
"""

name = "4h_Trix_ZeroCross_VolumeSpike_12hTrend"
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
    volume = prices['volume'].values
    
    # TRIX calculation (15-period EMA of EMA of EMA of log returns)
    # Step 1: log returns
    log_returns = np.diff(np.log(np.concatenate([[close[0]], close])))
    
    # Step 2: triple EMA
    ema1 = np.full_like(log_returns, np.nan)
    ema2 = np.full_like(log_returns, np.nan)
    ema3 = np.full_like(log_returns, np.nan)
    
    if len(log_returns) >= 15:
        # Initialize EMAs
        ema1[14] = np.mean(log_returns[0:15])
        ema2[14] = np.mean(ema1[0:15]) if not np.isnan(ema1[14]) else np.nan
        ema3[14] = np.mean(ema2[0:15]) if not np.isnan(ema2[14]) else np.nan
        
        # Calculate EMAs
        alpha = 2.0 / (15 + 1)
        for i in range(15, len(log_returns)):
            ema1[i] = alpha * log_returns[i] + (1 - alpha) * ema1[i-1]
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1] if not np.isnan(ema1[i]) else np.nan
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1] if not np.isnan(ema2[i]) else np.nan
    
    trix = ema3 * 100  # Scale for readability
    
    # Align TRIX to price array (shifted by 1 due to diff)
    trix_aligned = np.full_like(close, np.nan)
    trix_aligned[1:] = trix[:-1]  # Align with price (TRIX[i] corresponds to price[i+1])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        ema_34_12h[33] = np.mean(close_12h[0:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = (2.0 * close_12h[i] + (34 - 1) * ema_34_12h[i-1]) / (34 + 1)
    
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND uptrend (price > EMA34) AND volume spike
            if (i > 0 and not np.isnan(trix_aligned[i-1]) and 
                trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and
                close[i] > ema_34_12h_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND downtrend (price < EMA34) AND volume spike
            elif (i > 0 and not np.isnan(trix_aligned[i-1]) and 
                  trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR trend reversal (price < EMA34)
            if (i > 0 and not np.isnan(trix_aligned[i-1]) and 
                trix_aligned[i-1] >= 0 and trix_aligned[i] < 0) or \
               close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR trend reversal (price > EMA34)
            if (i > 0 and not np.isnan(trix_aligned[i-1]) and 
                trix_aligned[i-1] <= 0 and trix_aligned[i] > 0) or \
               close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals