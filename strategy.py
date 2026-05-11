#!/usr/bin/env python3
"""
6h_VolumeSpike_SupportResistance_Pullback
Hypothesis: In strong trending markets (1d EMA50 filter), price pulls back to key support/resistance levels (10-period SMA) with volume exhaustion signals. Enter on volume spikes in direction of trend after pullback. Works in bull/bear markets by following higher timeframe trend and using mean reversion within trend.
"""

name = "6h_VolumeSpike_SupportResistance_Pullback"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Indicators on 6h timeframe ===
    # 10-period SMA for pullback level
    sma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Volume spike: 2.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.5
    
    # Volume dry up: below 0.5x 20-period EMA (exhaustion signal)
    volume_dry = volume < vol_ema20 * 0.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers daily EMA50 and SMA10)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(sma10[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(volume_dry[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: pullback to support in uptrend with volume spike
            if (close[i] <= sma10[i] * 1.005 and  # near or slightly above SMA10
                close[i] > ema50_1d_aligned[i] and  # uptrend filter
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: pullback to resistance in downtrend with volume spike
            elif (close[i] >= sma10[i] * 0.995 and  # near or slightly below SMA10
                  close[i] < ema50_1d_aligned[i] and  # downtrend filter
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: volume dry up (exhaustion) or trend reversal
            if volume_dry[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: volume dry up (exhaustion) or trend reversal
            if volume_dry[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals