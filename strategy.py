#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams Alligator + 1d Elder Ray (Bull/Bear Power) + volume confirmation
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND 1d Bull Power > 0 AND volume > 1.5 * avg_volume(20)
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND 1d Bear Power < 0 AND volume > 1.5 * avg_volume(20)
# Exit when Alligator alignment reverses OR volume drops below avg_volume(20)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe
# Alligator (13,8,5 SMAs) identifies trend structure without lag
# Elder Ray measures bull/bear power relative to EMA13 for confirmation
# Volume confirmation filters low-conviction signals
# Works in bull markets (trend continuations) and bear markets (trend continuations down)

name = "6h_12hAlligator_1dElderRay_Volume"
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data ONCE before loop for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:  # Need at least 13 completed 12h bars for Alligator
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate Williams Alligator (13,8,5 SMAs) on 12h
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values   # Red line
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Align 12h Alligator levels to 6h timeframe (wait for completed 12h bar)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data ONCE before loop for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # Need at least 13 completed daily bars for EMA13
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    volume_normal = volume > avg_volume_20  # For exit condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator bullish alignment AND Bull Power positive AND volume spike
            if (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and  # Jaws < Teeth < Lips
                bull_power_aligned[i] > 0 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment AND Bear Power negative AND volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and  # Jaws > Teeth > Lips
                  bear_power_aligned[i] < 0 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment turns bearish OR volume drops to normal
            if not (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]) or not volume_normal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment turns bullish OR volume drops to normal
            if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) or not volume_normal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals