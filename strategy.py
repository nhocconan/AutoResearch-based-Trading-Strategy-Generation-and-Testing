#!/usr/bin/env python3
# 4h_CCI_Zero_Cross_12hTrend_VolumeSpike
# Hypothesis: CCI crossing zero indicates momentum shifts. When combined with 12h trend and volume spikes,
# it captures trend-following moves with lower whipsaw. Works in bull via long crossovers above zero with uptrend,
# and bear via short crossovers below zero with downtrend. Volume filter ensures institutional participation.
# Target: 25-40 trades per year (~100-160 over 4 years) with position size 0.25.

name = "4h_CCI_Zero_Cross_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA20 for trend filter
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # CCI(20) on 4h
    typical_price = (high + low + close) / 3.0
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = np.where(tp_mad != 0, (typical_price - tp_ma) / (0.015 * tp_mad), 0.0)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for CCI and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(cci[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # CCI zero cross signals
        cci_cross_up = cci[i-1] <= 0 and cci[i] > 0   # crosses above zero
        cci_cross_down = cci[i-1] >= 0 and cci[i] < 0  # crosses below zero
        
        # Volume confirmation: volume > 2x average
        volume_confirm = vol_ratio[i] > 2.0
        
        # Trend filter from 12h EMA20
        uptrend = close[i] > ema_20_12h_aligned[i]
        downtrend = close[i] < ema_20_12h_aligned[i]
        
        if position == 0:
            # Long: CCI crosses above zero + volume + uptrend
            if cci_cross_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: CCI crosses below zero + volume + downtrend
            elif cci_cross_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: CCI crosses back below zero or trend reversal
            if cci[i] < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: CCI crosses back above zero or trend reversal
            if cci[i] > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals