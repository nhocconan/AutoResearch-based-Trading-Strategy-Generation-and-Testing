#!/usr/bin/env python3
"""
4h Williams Alligator + Elder Ray + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend absence when intertwined; Elder Ray (Bull/Bear Power) measures trend strength; Volume Spike confirms institutional participation. Strategy enters long when alligator aligned bullish (lips>teeth>jaw), Bull Power positive and rising, Bear Power negative, and volume > 1.5x 20-bar average. Short when alligator aligned bearish (lips<teeth<jaw), Bear Power negative and falling, Bull Power positive, and volume spike. Uses 1d EMA34 as higher-timeframe trend filter to ensure alignment with daily trend. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-40 trades/year on 4h.
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
    
    # Get 1d data for EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator (13,8,5 smoothed with 8,5,3)
    # Jaw (13-period SMMA of median price, smoothed 8)
    median_price = (high + low) / 2
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth (8-period SMMA of median price, smoothed 5)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips (5-period SMMA of median price, smoothed 3)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Elder Ray (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation: volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator calculations (13+8=21, plus smoothing)
    start_idx = 25
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_vol_spike = volume_spike[i]
        ema_trend = ema_34_1d_aligned[i]
        
        # Alligator alignment
        alligator_bull = (curr_lips > curr_teeth) and (curr_teeth > curr_jaw)  # Lips > Teeth > Jaw
        alligator_bear = (curr_lips < curr_teeth) and (curr_teeth < curr_jaw)  # Lips < Teeth < Jaw
        
        if position == 0:
            # Look for entry signals
            # Long: Alligator bullish aligned + Bull Power > 0 and rising + Bear Power < 0 + Volume spike + Price > 1d EMA34
            bull_power_rising = (i > start_idx) and (curr_bull > bull_power[i-1])
            long_entry = alligator_bull and bull_power_rising and (curr_bear < 0) and curr_vol_spike and (curr_close > ema_trend)
            
            # Short: Alligator bearish aligned + Bear Power < 0 and falling + Bull Power > 0 + Volume spike + Price < 1d EMA34
            bear_power_falling = (i > start_idx) and (curr_bear < bear_power[i-1])
            short_entry = alligator_bear and bear_power_falling and (curr_bull > 0) and curr_vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator loses bullish alignment OR Bear Power becomes positive OR Volume drops
            if not alligator_bull or (curr_bear >= 0) or (not curr_vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator loses bearish alignment OR Bull Power becomes negative OR Volume drops
            if not alligator_bear or (curr_bull <= 0) or (not curr_vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_VolumeSpike_1dEMA34_Trend"
timeframe = "4h"
leverage = 1.0