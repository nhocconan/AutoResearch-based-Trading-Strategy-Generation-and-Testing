#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for trend direction, 1d Camarilla pivot levels (R1,S1) for reversal entries,
# and volume spikes (2x 20-period average) to confirm reversals. Works in both bull and bear
# markets by trading reversals against the 1d trend at key pivot levels. Target: 15-25 trades/year
# to minimize fee decay while capturing mean-reversion moves at institutional levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34) and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d for trend
    close_1d = df_1d['close'].values
    ema_len = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_1d[ema_len-1] = np.mean(close_1d[:ema_len])
        for i in range(ema_len, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate Camarilla pivot levels on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(high_1d)
    camarilla_r1 = np.full(n_1d, np.nan)
    camarilla_s1 = np.full(n_1d, np.nan)
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    for i in range(n_1d):
        h = high_1d[i]
        l = low_1d[i]
        c = close_1d[i]
        camarilla_r1[i] = c + (h - l) * 1.1 / 12
        camarilla_s1[i] = c - (h - l) * 1.1 / 12
    
    # Align 1d indicators to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 20-period average volume on 12h for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, 20)  # EMA34 needs 34, volume needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: at least 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: Price touches S1 with uptrend and volume (mean reversion long)
            if price <= camarilla_s1_aligned[i] and price > ema_1d_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Price touches R1 with downtrend and volume (mean reversion short)
            elif price >= camarilla_r1_aligned[i] and price < ema_1d_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price reaches midpoint or trend reversal
            midpoint = (camarilla_s1_aligned[i] + camarilla_r1_aligned[i]) / 2
            if price >= midpoint or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price reaches midpoint or trend reversal
            midpoint = (camarilla_s1_aligned[i] + camarilla_r1_aligned[i]) / 2
            if price <= midpoint or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0