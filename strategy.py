#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_elder_ray_power_v1
# Uses Elder Ray (Bull/Bear Power) from 1d timeframe with 60-period EMA.
# Bull Power = High - EMA60, Bear Power = EMA60 - Low.
# Long when Bull Power > 0 and rising, short when Bear Power > 0 and rising.
# Filters: volume > 1.5x 20-period average, avoids whipsaws in low volume.
# Target: 15-30 trades/year per symbol, works in both bull and bear via power shift.

name = "6h_1d_elder_ray_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 60-period EMA on daily close
    close_1d = df_1d['close'].values
    ema60 = pd.Series(close_1d).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Elder Ray components
    bull_power = df_1d['high'].values - ema60  # High - EMA60
    bear_power = ema60 - df_1d['low'].values   # EMA60 - Low
    
    # Align to 6h timeframe
    ema60_aligned = align_htf_to_ltf(prices, df_1d, ema60)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # start after EMA warmup
        # Skip if values not ready
        if np.isnan(ema60_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if no volume
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: Bull Power positive AND rising (momentum)
        if bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_aligned[i-1] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: Bear Power positive AND rising (momentum)
        elif bear_power_aligned[i] > 0 and bear_power_aligned[i] > bear_power_aligned[i-1] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: when power turns negative (loss of momentum)
        elif position == 1 and bull_power_aligned[i] <= 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bear_power_aligned[i] <= 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals