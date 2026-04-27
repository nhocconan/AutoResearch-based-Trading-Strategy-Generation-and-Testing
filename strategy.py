#!/usr/bin/env python3
"""
6h_RollingVolume_Spike_1wTrend_Filter
Hypothesis: On 6h, a volume spike (2x 20-period average) in the direction of 1w trend
(captured by 1w EMA20) captures strong moves while avoiding chop. Weekly trend filter
adapts to bull/bear markets. Target: 15-25 trades/year to minimize fee drag.
"""

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
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w trend: EMA20
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume spike: volume > 2 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema20_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema20_1w_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: volume spike + price above 1w EMA (uptrend)
            if vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: volume spike + price below 1w EMA (downtrend)
            elif vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1w EMA
            if close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1w EMA
            if close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RollingVolume_Spike_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0