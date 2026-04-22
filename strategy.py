#!/usr/bin/env python3

"""
Hypothesis: 4-hour Williams Alligator with 12-hour trend filter and volume confirmation.
The Williams Alligator (Jaw/Teeth/Lips) identifies trend presence and direction.
Using 12-hour trend filter (price above/below EMA50) avoids counter-trend trades.
Volume spikes confirm institutional interest. This combination should work in both
bull and bear regimes by aligning with the higher timeframe trend.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator lines"""
    # Jaw (Blue): 13-period SMMA smoothed 8 bars ahead
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean()
    jaw = jaw.shift(8)
    
    # Teeth (Red): 8-period SMMA smoothed 5 bars ahead
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean()
    teeth = teeth.shift(5)
    
    # Lips (Green): 5-period SMMA smoothed 3 bars ahead
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean()
    lips = lips.shift(3)
    
    return jaw.values, teeth.values, lips.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'])
    ema_50_12h = close_12h.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Determine 12h trend: price above/below EMA50
    bullish_trend_12h = df_12h['close'].values > ema_50_12h
    bearish_trend_12h = df_12h['close'].values < ema_50_12h
    
    # Align 12h trend to 4h timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_12h, bullish_trend_12h.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_12h, bearish_trend_12h.astype(float))
    
    # Calculate 4h Williams Alligator
    jaw, teeth, lips = calculate_alligator(high, low, close, 13, 8, 5)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment), bullish 12h trend, volume spike
            if (lips[i] > teeth[i] > jaw[i] and
                bullish_aligned[i] > 0.5 and
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment), bearish 12h trend, volume spike
            elif (lips[i] < teeth[i] < jaw[i] and
                  bearish_aligned[i] > 0.5 and
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines cross in opposite direction
            exit_signal = False
            
            if position == 1:
                # Exit long: Lips < Teeth (bullish alignment broken)
                if lips[i] < teeth[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Lips > Teeth (bearish alignment broken)
                if lips[i] > teeth[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0