#/usr/bin/env python3

"""
Hypothesis: 12-hour Williams Alligator system with 1-day EMA(34) trend filter and volume confirmation.
Trades in direction of Alligator alignment only when daily trend agrees and volume exceeds 1.5x average.
Targets 12-30 trades/year (48-120 total over 4 years) to minimize fee drag.
Williams Alligator uses smoothed medians (Jaw/Teeth/Lips) to identify trends and avoid whipsaws.
Works in trending markets by following the Alligator's mouth direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    sma = np.mean(data[:period])
    result[period-1] = sma
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (13,8,5 periods)
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    jaw = smma(median_12h, 13)  # Blue line
    teeth = smma(median_12h, 8)  # Red line
    lips = smma(median_12h, 5)   # Green line
    
    # Align Alligator lines
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: Alligator aligned upward (Lips > Teeth > Jaw) AND price above daily EMA (uptrend)
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned downward (Lips < Teeth < Jaw) AND price below daily EMA (downtrend)
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator alignment breaks or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator alignment breaks downward OR price closes below daily EMA
                if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator alignment breaks upward OR price closes above daily EMA
                if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0