#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_RegimeFilter_VolumeSpike
Hypothesis: Camarilla R1/S1 breakouts with volume spike and choppiness regime filter work in both bull and bear markets.
In bull: breakouts continue trends. In bear: choppiness filter avoids false breakouts, mean reversion at pivots provides short opportunities.
Uses 4h for execution, 1d for pivots and chop filter, 12h for trend confirmation. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / log10(range)) / log10(14)
    # High CHOP (>61.8) = ranging/choppy, Low CHOP (<38.2) = trending
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_14 * 14 / (max_high - min_low + 1e-10)) / np.log10(14)
    chop_raw = np.where((max_high - min_low) > 0, chop_raw, 50.0)  # neutral when range=0
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    uptrend_12h = close > ema_50_12h_aligned
    downtrend_12h = close < ema_50_12h_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA + 14 for chop + 50 for EMA)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: close breaks above R1, with 12h uptrend, volume spike, and NOT too choppy (CHOP < 61.8)
            if (close[i] > camarilla_r1_aligned[i] and uptrend_12h[i] and 
                volume_spike[i] and chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S1, with 12h downtrend, volume spike, and NOT too choppy (CHOP < 61.8)
            elif (close[i] < camarilla_s1_aligned[i] and downtrend_12h[i] and 
                  volume_spike[i] and chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close drops below S1 (mean reversion) OR 12h trend changes to downtrend OR chop becomes too high
            if (close[i] < camarilla_s1_aligned[i] or not uptrend_12h[i] or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close rises above R1 (mean reversion) OR 12h trend changes to uptrend OR chop becomes too high
            if (close[i] > camarilla_r1_aligned[i] or not downtrend_12h[i] or chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_RegimeFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0