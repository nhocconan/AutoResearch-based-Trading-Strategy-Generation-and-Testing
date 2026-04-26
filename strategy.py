#!/usr/bin/env python3
"""
6h_ElderRay_ZeroCross_12hTrend_VolumeSpike_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) zero-cross with 12h EMA50 trend filter and volume spike (>1.8x median). 
Bull Power = Close - EMA13, Bear Power = EMA13 - Close. 
Long when Bull Power crosses above zero (bullish momentum) in uptrend with volume confirmation.
Short when Bear Power crosses above zero (bearish momentum) in downtrend with volume confirmation.
Uses 12h timeframe for HTF trend to reduce noise and avoid overtrading (~25-40 trades/year).
Designed to work in both bull and bear markets by only trading with the 12h trend direction.
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
    
    # Get 12h data for HTF trend (EMA50)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = close - ema_13  # Bull Power: Close - EMA13
    bear_power = ema_13 - close  # Bear Power: EMA13 - Close
    
    # Zero-cross signals: previous value <= 0, current value > 0
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = -np.inf  # Ensure no false signal on first bar
    bear_power_prev[0] = -np.inf
    
    bull_zero_cross = (bull_power_prev <= 0) & (bull_power > 0)
    bear_zero_cross = (bear_power_prev <= 0) & (bear_power > 0)
    
    # Volume spike filter: volume > 1.8x median volume (30-period)
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(50) 12h, EMA(13), Bull/Bear Power, volume median (30)
    start_idx = max(50, 13, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_12h_val = ema_50_12h_aligned[i]
        close_val = close[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        bull_cross = bull_zero_cross[i]
        bear_cross = bear_zero_cross[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_12h_val
        downtrend = close_val < ema_50_12h_val
        
        # Volume confirmation
        volume_spike = volume_val > 1.8 * vol_median_val
        
        if position == 0:
            # Long: Bull Power crosses above zero with volume spike, and uptrend
            long_signal = bull_cross and \
                          volume_spike and \
                          uptrend
            
            # Short: Bear Power crosses above zero with volume spike, and downtrend
            short_signal = bear_cross and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit when Bull Power crosses below zero (momentum loss) OR trend change
            if bull_power_val <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit when Bear Power crosses below zero (momentum loss) OR trend change
            if bear_power_val <= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroCross_12hTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0