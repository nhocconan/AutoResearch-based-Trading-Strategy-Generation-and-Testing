#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_Trend_Filter
# Hypothesis: TRIX (triple exponential average momentum) on 4h detects momentum shifts.
# Combined with 1d EMA trend filter and volume spike to confirm direction.
# Long when TRIX crosses above zero + 1d uptrend + volume spike.
# Short when TRIX crosses below zero + 1d downtrend + volume spike.
# Designed for low-to-moderate trade frequency (target: 20-50 trades/year) with strong momentum signals.

name = "4h_TRIX_VolumeSpike_Trend_Filter"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate TRIX on 4h close (15-period)
    # EMA1 = EMA(close, 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).values
    # EMA2 = EMA(EMA1, 15)
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).values
    # EMA3 = EMA(EMA2, 15)
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).values
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix = np.zeros(n)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix[0] = 0
    
    # Calculate daily EMA for trend filter (34-period)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA on 4h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX (~15*3 for EMA chain), EMA34, volume MA
    start_idx = max(45, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # TRIX zero-cross signals
        trix_cross_up = trix[i-1] <= 0 and trix[i] > 0
        trix_cross_down = trix[i-1] >= 0 and trix[i] < 0
        
        if position == 0:
            # Long entry: TRIX crosses up + daily uptrend + volume spike
            if trix_cross_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses down + daily downtrend + volume spike
            elif trix_cross_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero or daily trend turns down
            if trix[i] < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero or daily trend turns up
            if trix[i] > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals