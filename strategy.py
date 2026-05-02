#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power combination
# Uses 1d HTF for Elder Ray (Bull/Bear Power) to determine market regime
# 6h Williams Alligator (Jaw=13, Teeth=8, Lips=5) for trend direction and entry timing
# Long when: Bull Power > 0, Bear Power < 0, and price > Alligator Teeth (8) AND price > Alligator Lips (5)
# Short when: Bull Power < 0, Bear Power > 0, and price < Alligator Teeth (8) AND price < Alligator Lips (5)
# Volume confirmation: 1.5x 20-period average to ensure participation
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Discrete sizing 0.25 to balance return and fee drag
# Target: 80-120 total trades over 4 years (20-30/year) to minimize fee impact while capturing regime shifts

name = "6h_WilliamsAlligator_ElderRay_Power_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Williams Alligator on 6h timeframe
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Calculate SMMA using EMA with adjusted alpha (SMMA ≈ EMA with alpha=1/period)
    jaw = pd.Series(close).ewm(alpha=1/jaw_period, adjust=False, min_periods=jaw_period).mean().values
    teeth = pd.Series(close).ewm(alpha=1/teeth_period, adjust=False, min_periods=teeth_period).mean().values
    lips = pd.Series(close).ewm(alpha=1/lips_period, adjust=False, min_periods=lips_period).mean().values
    
    # Calculate Elder Ray Power from 1d HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA of 1d close for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13(close)
    bull_power = high_1d - ema_13_1d
    # Bear Power = Low - EMA13(close)
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray to 6h timeframe (wait for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(jaw_period, teeth_period, lips_period, 13) + 20
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0, Bear Power < 0, price > Teeth AND price > Lips, volume spike
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                close[i] > teeth[i] and 
                close[i] > lips[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, price < Teeth AND price < Lips, volume spike
            elif (bull_power_aligned[i] < 0 and 
                  bear_power_aligned[i] > 0 and 
                  close[i] < teeth[i] and 
                  close[i] < lips[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 OR Bear Power >= 0 OR price < Lips
            if (bull_power_aligned[i] <= 0 or 
                bear_power_aligned[i] >= 0 or 
                close[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull Power >= 0 OR Bear Power <= 0 OR price > Lips
            if (bull_power_aligned[i] >= 0 or 
                bear_power_aligned[i] <= 0 or 
                close[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals