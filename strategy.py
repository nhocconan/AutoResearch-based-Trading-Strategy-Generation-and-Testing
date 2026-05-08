#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Angle_of_Attack_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for angle of attack and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily close for calculations
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Angle of attack: 45-degree angle from 10-period low
    # Calculate lowest low of last 10 days
    low_min10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    # Calculate the angle from that low to current close
    # Angle = arctan((close - low_min10) / 10) * 180/pi
    # We use tangent directly to avoid trig functions: (close - low_min10) / 10
    angle_of_attack = (close_1d - low_min10) / 10.0
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = (close_1d > ema50_1d).astype(float)
    
    # Daily volume spike: current volume > 2.0 * 20-day average
    vol_ma20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma20d * 2.0)
    
    # Align all daily indicators to 6h timeframe
    angle_aligned = align_htf_to_ltf(prices, df_1d, angle_of_attack)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(angle_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: strong upward angle (>0.5) with volume spike and daily uptrend
            long_cond = (angle_aligned[i] > 0.5 and vol_spike_aligned[i] and trend_aligned[i] > 0.5)
            
            # Short entry: strong downward angle (<-0.5) with volume spike and daily downtrend
            short_cond = (angle_aligned[i] < -0.5 and vol_spike_aligned[i] and trend_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: angle turns negative or volume dries up
            if angle_aligned[i] < 0.0 or not vol_spike_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: angle turns positive or volume dries up
            if angle_aligned[i] > 0.0 or not vol_spike_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Angle of attack measures the steepness of the daily trend from recent lows.
# In bull markets: strong upward angles signal continuation; in bear markets: strong downward angles signal continuation.
# Volume spike confirms institutional participation; daily EMA50 ensures alignment with longer-term trend.
# Target: 15-30 trades/year to minimize fee decay while capturing strong trending moves.