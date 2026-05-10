#!/usr/bin/env python3
# 4h_TRIX_Threshold_VolumeSpike_Cross_1dTrend_v1
# Hypothesis: TRIX on 4h combined with daily trend filter and volume spikes provides
# robust momentum signals. TRIX (12) crossing above zero in a daily uptrend
# with volume > 2x 20-bar average signals long entries; crossing below zero in
# a daily downtrend signals short entries. This combination filters whipsaws
# and captures sustained trends in both bull and bear markets. Exit when TRIX
# crosses back through zero or volume drops below threshold.

name = "4h_TRIX_Threshold_VolumeSpike_Cross_1dTrend_v1"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # TRIX on 4h: EMA(EMA(EMA(close,12),12),12) then percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = 100 * (pd.Series(ema3).pct_change())
    trix = trix_raw.fillna(0).values
    
    # Volume confirmation (20-period MA on 4h = ~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA34 (34), TRIX (~36), volume MA (20)
    start_idx = max(34, 36, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(trix[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (>2.0x MA to reduce false signals)
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: TRIX crosses above zero in uptrend with volume
            if trix[i] > 0 and trix[i-1] <= 0 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero in downtrend with volume
            elif trix[i] < 0 and trix[i-1] >= 0 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero or volume drops
            if trix[i] < 0 or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero or volume drops
            if trix[i] > 0 or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals