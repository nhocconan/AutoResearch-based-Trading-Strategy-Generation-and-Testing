#!/usr/bin/env python3
"""
1d_TRIX_9_VolumeSpike_1wTrend
Hypothesis: TRIX(9) zero-cross on 1d with 1w EMA50 trend filter and volume spike confirmation.
TRIX is a triple-smoothed EMA momentum oscillator that filters noise and identifies trend changes.
In bear markets, TRIX zero-cross from below with volume spike and 1w trend up captures relief rallies.
In bull markets, TRIX zero-cross from above with volume spike and 1w trend down captures pullbacks.
Designed for 10-25 trades/year on 1d to minimize fee drag while maintaining edge in both regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX(9): triple EMA of close, then ROC
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean()
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean()
    trix = 100 * (pd.Series(ema3).pct_change())
    trix_values = trix.values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for TRIX (3*9=27), EMA50, volume average
    start_idx = max(50, 30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_values[i]) or np.isnan(trix_values[i-1]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        trix_now = trix_values[i]
        trix_prev = trix_values[i-1]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        
        if position == 0:
            # Flat - look for entry: TRIX zero-cross with volume spike and 1w trend alignment
            # Long: TRIX crosses above zero AND 1w trend is up (close > EMA50) AND volume spike
            # Short: TRIX crosses below zero AND 1w trend is down (close < EMA50) AND volume spike
            long_cross = trix_prev <= 0 and trix_now > 0
            short_cross = trix_prev >= 0 and trix_now < 0
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            
            if long_cross and trend_up and vol_spike:
                signals[i] = size
                position = 1
            elif short_cross and trend_down and vol_spike:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long - exit when TRIX crosses below zero (momentum loss) OR 1w trend turns down
            if trix_now < 0 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TRIX crosses above zero (momentum loss) OR 1w trend turns up
            if trix_now > 0 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_TRIX_9_VolumeSpike_1wTrend"
timeframe = "1d"
leverage = 1.0