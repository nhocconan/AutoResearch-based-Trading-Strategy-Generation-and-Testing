#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with Elder Ray power and volume spike.
# Uses 4h Alligator (SMAs) for trend direction and 1h Elder Ray for momentum.
# Enters only during 08-20 UTC session to avoid low-volume noise.
# Targets 20-50 trades/year (80-200 total over 4 years) with strict entry conditions.
# Works in bull/bear by following higher timeframe trends and momentum.
name = "4h_WilliamsAlligator_ElderRay_Volume_Spike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Williams Alligator (SMAs: 13, 8, 5)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Alligator Jaw (13-period SMA)
    jaw_13_4h = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().values
    # Alligator Teeth (8-period SMA)
    teeth_8_4h = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().values
    # Alligator Lips (5-period SMA)
    lips_5_4h = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().values
    jaw_13_4h_aligned = align_htf_to_ltf(prices, df_4h, jaw_13_4h)
    teeth_8_4h_aligned = align_htf_to_ltf(prices, df_4h, teeth_8_4h)
    lips_5_4h_aligned = align_htf_to_ltf(prices, df_4h, lips_5_4h)
    
    # Get 1h data for Elder Ray (requires 13-period EMA)
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_1h = pd.Series(close_1h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1h = high_1h - ema_13_1h
    bear_power_1h = low_1h - ema_13_1h
    bull_power_1h_aligned = align_htf_to_ltf(prices, df_1h, bull_power_1h)
    bear_power_1h_aligned = align_htf_to_ltf(prices, df_1h, bear_power_1h)
    
    # Volume filter: volume > 2.0 * 20-period average (4h)
    volume_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_4h * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(jaw_13_4h_aligned[i]) or np.isnan(teeth_8_4h_aligned[i]) or 
            np.isnan(lips_5_4h_aligned[i]) or np.isnan(bull_power_1h_aligned[i]) or
            np.isnan(bear_power_1h_aligned[i]) or np.isnan(volume_ma_4h[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND volume spike
            if (lips_5_4h_aligned[i] > teeth_8_4h_aligned[i] > jaw_13_4h_aligned[i] and 
                bull_power_1h_aligned[i] > 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) AND Bear Power < 0 AND volume spike
            elif (jaw_13_4h_aligned[i] > teeth_8_4h_aligned[i] > lips_5_4h_aligned[i] and 
                  bear_power_1h_aligned[i] < 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator alignment breaks or Bull Power <= 0
            if (lips_5_4h_aligned[i] <= teeth_8_4h_aligned[i] or 
                teeth_8_4h_aligned[i] <= jaw_13_4h_aligned[i] or
                bull_power_1h_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator alignment breaks or Bear Power >= 0
            if (jaw_13_4h_aligned[i] <= teeth_8_4h_aligned[i] or 
                teeth_8_4h_aligned[i] <= lips_5_4h_aligned[i] or
                bear_power_1h_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals