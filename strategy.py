#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
Long when price > Alligator's Jaw with bullish 1d trend and volume spike.
Short when price < Alligator's Jaw with bearish 1d trend and volume spike.
Exit when price crosses back below/above Jaw or trend reverses.
Williams Alligator uses SMAs of 13, 8, 5 periods shifted forward by 8, 5, 3 bars.
Designed for low trade frequency (15-25/year) to minimize fee drift in 12h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h
    close_12h = pd.Series(df_12h['close'].values)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw_12h = close_12h.rolling(window=13, min_periods=13).mean()
    jaw_12h = jaw_12h.shift(8)
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth_12h = close_12h.rolling(window=8, min_periods=8).mean()
    teeth_12h = teeth_12h.shift(5)
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips_12h = close_12h.rolling(window=5, min_periods=5).mean()
    lips_12h = lips_12h.shift(3)
    
    # Jaw is the main trend indicator
    jaw_12h_values = jaw_12h.values
    
    # Align Jaw to 12h timeframe (we're trading on 12h)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h_values)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume lookback
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(ema50_aligned[i]) or 
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
            # Long: Price > Jaw with bullish 1d trend and volume spike
            if (close[i] > jaw_aligned[i] and 
                close[i] > ema50_aligned[i] and  # Bullish trend: price above EMA50
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price < Jaw with bearish 1d trend and volume spike
            elif (close[i] < jaw_aligned[i] and 
                  close[i] < ema50_aligned[i] and  # Bearish trend: price below EMA50
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Jaw OR trend turns bearish
                if close[i] <= jaw_aligned[i] or close[i] < ema50_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Jaw OR trend turns bullish
                if close[i] >= jaw_aligned[i] or close[i] > ema50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0
#%%