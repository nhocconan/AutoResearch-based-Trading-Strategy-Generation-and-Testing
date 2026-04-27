#!/usr/bin/env python3
"""
12h_WilliamsAlligator_ElderRay_TrendFilter
Hypothesis: Williams Alligator (Jaws/Teeth/Lips) identifies trend direction, Elder Ray (Bull/Bear Power) confirms momentum strength. Long when price > Teeth and Bull Power > 0; Short when price < Teeth and Bear Power < 0. Uses 1d EMA13 for trend filter and volume spike for confirmation. Works in bull via Alligator alignment and bear via divergence. Targets ~20 trades/year on 12h to minimize fee drag.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for trend filter
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    # Jaws: SMA(13, 8) - 13-period SMA shifted 8 bars ahead
    # Teeth: SMA(8, 5) - 8-period SMA shifted 5 bars ahead
    # Lips: SMA(5, 3) - 5-period SMA shifted 3 bars ahead
    jaws_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Shift to avoid look-ahead (Williams Alligator uses future values)
    jaws = np.roll(jaws_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set initial values to NaN due to roll
    jaws[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    # Volume confirmation: volume > 1.5 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Alligator and volume MA
    start_idx = max(13, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema13_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw = jaws_aligned[i]
        tooth = teeth_aligned[i]
        lip = lips_aligned[i]
        ema_trend = ema13_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price > Teeth, Bull Power > 0, volume spike, and uptrend alignment
            if close[i] > tooth and bull > 0 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price < Teeth, Bear Power < 0, volume spike, and downtrend alignment
            elif close[i] < tooth and bear < 0 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < Lips or Bear Power < 0 or trend turns down
            if close[i] < lip or bear < 0 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price > Lips or Bull Power > 0 or trend turns up
            if close[i] > lip or bull > 0 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_TrendFilter"
timeframe = "12h"
leverage = 1.0