#!/usr/bin/env python3
# 4h_1d_ema_trend_volume_v1
# Hypothesis: 4h EMA(21) trend following with 1d EMA(50) confirmation and volume spikes.
# Works in bull markets (EMA21 > EMA50 with volume) and bear markets (EMA21 < EMA50 with volume).
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) by requiring multi-timeframe alignment and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA(21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Get 1d data for EMA(50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(21)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.8x average of last 96 periods (24 hours)
    vol_ma = pd.Series(volume).rolling(window=96, min_periods=96).mean().values
    vol_confirm = volume > vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: 4h EMA crosses below 1d EMA
            if ema_4h_aligned[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: 4h EMA crosses above 1d EMA
            if ema_4h_aligned[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: 4h EMA above 1d EMA with volume
            if ema_4h_aligned[i] > ema_1d_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: 4h EMA below 1d EMA with volume
            elif ema_4h_aligned[i] < ema_1d_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals