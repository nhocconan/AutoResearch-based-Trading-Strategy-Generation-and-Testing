#!/usr/bin/env python3
"""
12h_TRIX_0_VolumeSpike_PhaseFilter
Hypothesis: TRIX crossing zero indicates momentum shift. Combine with volume spike (>1.5x 20-period avg) and price above/below 1w EMA50 for trend filter. Long when TRIX crosses above zero in uptrend with volume spike. Short when TRIX crosses below zero in downtrend with volume spike. Uses 12h primary timeframe to limit trades and reduce fee drag. Works in both bull and bear markets by following momentum shifts confirmed by volume and trend.
"""

name = "12h_TRIX_0_VolumeSpike_PhaseFilter"
timeframe = "12h"
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
    
    # === 1W Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA50 on 1w close for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === TRIX Calculation (12-period EMA of 12-period EMA of 12-period EMA of price) ===
    # TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3_prev = np.roll(ema3, 1)
    ema3_prev[0] = ema3[0]
    trix = 100 * (ema3 - ema3_prev) / ema3_prev
    
    # === Volume Spike: current volume > 1.5x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for TRIX and EMA
    start_idx = 36
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema3[i]) or 
            np.isnan(ema3_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero AND uptrend (price > 1w EMA50) AND volume spike
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero AND downtrend (price < 1w EMA50) AND volume spike
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero OR price crosses below 1w EMA50
            if trix[i] < 0 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: TRIX crosses above zero OR price crosses above 1w EMA50
            if trix[i] > 0 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals