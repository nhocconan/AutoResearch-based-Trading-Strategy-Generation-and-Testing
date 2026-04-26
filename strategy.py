#!/usr/bin/env python3
"""
6h_ElderRay_Alligator_1dTrend_v1
Hypothesis: Combine Elder Ray (Bull/Bear Power) with Williams Alligator on 6h timeframe, filtered by 1d EMA50 trend. 
Elder Ray measures bull/bear power relative to EMA13; Alligator (JAW/TEETH/LIPS) confirms trend alignment. 
Only take longs when Bull Power > 0, Bear Power < 0, and price above Alligator teeth (red line) in uptrend (1d EMA50 up).
Only take shorts when Bear Power < 0, Bull Power < 0, and price below Alligator teeth in downtrend.
Uses volume confirmation (1.5x median) to avoid false signals. Designed for low-frequency, high-conviction trades 
that work in both bull (trend following) and bear (counter-trend reversals via Elder Ray extremes) markets.
Target: 12-30 trades/year (50-120 over 4 years) with discrete sizing 0.25 to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h EMA(13) for Elder Ray foundation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Williams Alligator on 6h: SMAs with specific periods
    # JAW (Blue): 13-period SMMA shifted 8 bars ahead
    # TEETH (Red): 8-period SMMA shifted 5 bars ahead  
    # LIPS (Green): 5-period SMMA shifted 3 bars ahead
    # Using regular SMA with shift for simplicity (SMMA approximation)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: 1.5x median volume
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1d EMA50 (50), EMA13 (13), Alligator JAW (13+8=21), volume median (30)
    start_idx = max(50, 13, 21, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        ema_13_val = ema_13[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        # Determine Alligator alignment: price > teeth = bullish alignment
        price_above_teeth = close_val > teeth_val
        price_below_teeth = close_val < teeth_val
        
        # Determine 1d trend: rising EMA50 = uptrend, falling = downtrend
        if i > start_idx:
            ema_50_1d_prev = ema_50_1d_aligned[i-1]
            uptrend_1d = ema_50_1d_val > ema_50_1d_prev
            downtrend_1d = ema_50_1d_val < ema_50_1d_prev
        else:
            uptrend_1d = False
            downtrend_1d = False
        
        if position == 0:
            # Long conditions:
            # 1. Bull Power > 0 (bulls in control)
            # 2. Bear Power < 0 (bears weak)
            # 3. Price above Alligator teeth (bullish alignment)
            # 4. 1d uptrend (higher timeframe bias)
            # 5. Volume confirmation
            long_signal = (bull_power_val > 0) and \
                          (bear_power_val < 0) and \
                          price_above_teeth and \
                          uptrend_1d and \
                          (volume_val > 1.5 * vol_median_val)
            
            # Short conditions:
            # 1. Bear Power < 0 (bears in control)
            # 2. Bull Power < 0 (bulls weak) 
            # 3. Price below Alligator teeth (bearish alignment)
            # 4. 1d downtrend (higher timeframe bias)
            # 5. Volume confirmation
            short_signal = (bear_power_val < 0) and \
                           (bull_power_val < 0) and \
                           price_below_teeth and \
                           downtrend_1d and \
                           (volume_val > 1.5 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period to reduce churn
            bars_since_entry += 1
            signals[i] = 0.25
            # Exit conditions:
            # 1. Minimum holding period (3 bars) AND
            # 2. Either: Bear Power turns negative (bulls losing) OR price breaks below teeth
            if bars_since_entry >= 3 and ((bear_power_val >= 0) or (close_val < teeth_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Exit conditions:
            # 1. Minimum holding period (3 bars) AND
            # 2. Either: Bull Power turns positive (bears losing) OR price breaks above teeth
            if bars_since_entry >= 3 and ((bull_power_val <= 0) or (close_val > teeth_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_Alligator_1dTrend_v1"
timeframe = "6h"
leverage = 1.0