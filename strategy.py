#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray combination with 1w trend filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for major trend direction (avoid counter-trend trades in bear markets).
- Williams Alligator (jaw=13, teeth=8, lips=5) identifies market phases:
  * Alligator sleeping (lines intertwined) = ranging market → fade extremes
  * Alligator awakening (lines separating) = trending → trade breakouts
- Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures trend strength:
  * Bull Power > 0 and rising = bullish momentum
  * Bear Power < 0 and falling = bearish momentum
- Entry logic:
  * Ranging market (Alligator asleep): Long when Bull Power crosses above 0 with rising momentum;
                                   Short when Bear Power crosses below 0 with falling momentum
  * Trending market (Alligator awake): Trade in direction of 1w trend only on pullbacks to EMA13
- Exit: Reverse signal or Alligator re-enters sleeping phase
- Signal size: 0.25 discrete to minimize fee drag
- Works in bull markets (buy the dip in uptrend, fade rallies in range) and bear markets 
  (sell the rally in downtrend, fade dips in range) with 1w trend filter preventing 
  counter-trend trades during major bear trends like 2022.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for major trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope_1w = ema_50_1w - np.roll(ema_50_1w, 1)
    ema_50_slope_1w[0] = 0
    # Align 1w trend to 6h
    ema_50_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_slope_1w)
    
    # Calculate Williams Alligator (13,8,5) on 6h data
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # Using EMA as approximation for SMMA (similar smoothing)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Alligator sleeping condition: lines are intertwined (max-min < threshold)
    # Normalize by ATR-like measure to make it adaptive
    atr_approx = pd.Series(high - low).rolling(14, min_periods=14).mean().values
    alligator_range = np.maximum(np.maximum(jaw, teeth), lips) - np.minimum(np.minimum(jaw, teeth), lips)
    sleeping_threshold = 0.5 * atr_approx  # Adaptive threshold
    alligator_sleeping = alligator_range < sleeping_threshold
    alligator_awake = ~alligator_sleeping
    
    # Calculate Elder Ray Power (using EMA13 as reference)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13   # Bull Power = High - EMA13
    bear_power = low - ema13    # Bear Power = Low - EMA13
    
    # Momentum of Elder Ray (rate of change)
    bull_power_momentum = bull_power - np.roll(bull_power, 1)
    bear_power_momentum = bear_power - np.roll(bear_power, 1)
    bull_power_momentum[0] = 0
    bear_power_momentum[0] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50(1w) and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_slope_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_momentum[i]) or np.isnan(bear_power_momentum[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Exit conditions
        if position != 0:
            # Exit if Alligator returns to sleeping phase (market losing momentum)
            if alligator_sleeping[i]:
                signals[i] = 0.0
                position = 0
                continue
            # Exit if 1w trend strongly opposes position
            if position == 1 and ema_50_slope_1w_aligned[i] < -0.001:  # Strong downtrend
                signals[i] = 0.0
                position = 0
                continue
            if position == -1 and ema_50_slope_1w_aligned[i] > 0.001:  # Strong uptrend
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market state
        is_sleeping = alligator_sleeping[i]
        is_awake = alligator_awake[i]
        
        # 1w trend direction
        uptrend_1w = ema_50_slope_1w_aligned[i] > 0.0001
        downtrend_1w = ema_50_slope_1w_aligned[i] < -0.0001
        
        if position == 0:
            if is_sleeping:
                # Ranging market: fade extremes using Elder Ray
                # Long: Bull Power crosses above 0 with rising momentum
                if bull_power[i] > 0 and bull_power_momentum[i] > 0 and bull_power[i-1] <= 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power crosses below 0 with falling momentum
                elif bear_power[i] < 0 and bear_power_momentum[i] < 0 and bear_power[i-1] >= 0:
                    signals[i] = -0.25
                    position = -1
            elif is_awake:
                # Trending market: trade pullbacks in 1w trend direction
                if uptrend_1w:
                    # Long on pullback to EMA13 (bear power rising from negative)
                    if bear_power[i] < 0 and bear_power_momentum[i] > 0 and bear_power[i-1] > bear_power[i]:
                        signals[i] = 0.25
                        position = 1
                elif downtrend_1w:
                    # Short on rally to EMA13 (bull power falling from positive)
                    if bull_power[i] > 0 and bull_power_momentum[i] < 0 and bull_power[i-1] < bull_power[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Maintain long position
            signals[i] = 0.25
            # Optional: exit if bull power deteriorates significantly
            if bull_power[i] < -0.5 * (high[i] - low[i]):  # Weak bull power
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Maintain short position
            signals[i] = -0.25
            # Optional: exit if bear power deteriorates significantly
            if bear_power[i] > 0.5 * (high[i] - low[i]):  # Weak bear power
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_1wTrendFilter_v1"
timeframe = "6h"
leverage = 1.0