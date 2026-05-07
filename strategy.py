#!/usr/bin/env python3
# 1d_TRIX_VolumeSpike_WedgeBreakout
# Hypothesis: Uses TRIX (15,9) momentum on 1d timeframe combined with volume spike confirmation.
# Long when TRIX crosses above zero with volume spike; short when TRIX crosses below zero with volume spike.
# Exit when TRIX crosses back across zero in opposite direction.
# TRIX helps filter noise and captures momentum shifts; volume spike confirms institutional participation.
# Designed for 1d to maintain low trade frequency (target 15-25/year) and work in both bull/bear markets
# by capturing momentum reversals. Uses 1w trend filter to avoid counter-trend trades in strong trends.

name = "1d_TRIX_VolumeSpike_WedgeBreakout"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate TRIX on 1d: TRIX = EMA(EMA(EMA(close, 15), 15), 15) then % change
    # Using 15-period triple EMA as standard
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 100 * (ema3 - previous ema3) / previous ema3
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0  # first value has no previous
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w trend to 1d timeframe
    ema_34_1w_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix_raw[i]) or np.isnan(ema_34_1w_1d[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + above 1w EMA34 trend + volume spike
            if trix_raw[i] > 0 and trix_raw[i-1] <= 0 and close[i] > ema_34_1w_1d[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + below 1w EMA34 trend + volume spike
            elif trix_raw[i] < 0 and trix_raw[i-1] >= 0 and close[i] < ema_34_1w_1d[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero (momentum reversal)
            if trix_raw[i] < 0 and trix_raw[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero (momentum reversal)
            if trix_raw[i] > 0 and trix_raw[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals