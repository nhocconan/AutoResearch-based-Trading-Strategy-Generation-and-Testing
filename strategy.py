#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1-week EMA50 trend filter and volume spike.
# Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 and rising,
# Bear Power < 0 and falling, with 1-week uptrend and volume spike. Reverse for short.
# Uses 13-period EMA for power calculation (standard) and 50-period EMA on 1-week for trend.
# Volume filter: current volume > 1.5x 20-period average to avoid chop.
# Designed for 10-25 trades/year per symbol (40-100 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1-week trend and requiring momentum confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA13 for Elder Ray power (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 50-period EMA on 1-week close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Bull Power > 0 and rising, Bear Power < 0 and falling,
        # 1-week uptrend, volume spike
        if (bull_power[i] > 0 and 
            bull_power[i] > bull_power[i-1] and 
            bear_power[i] < 0 and 
            bear_power[i] < bear_power[i-1] and 
            close[i] > ema50_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Bull Power < 0 and falling, Bear Power > 0 and rising,
        # 1-week downtrend, volume spike
        elif (bull_power[i] < 0 and 
              bull_power[i] < bull_power[i-1] and 
              bear_power[i] > 0 and 
              bear_power[i] > bear_power[i-1] and 
              close[i] < ema50_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0