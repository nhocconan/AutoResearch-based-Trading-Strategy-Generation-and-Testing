#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trending vs ranging markets
# 1d EMA50 ensures alignment with long-term trend to avoid counter-trend trades
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Designed for 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Works in bull markets (Alligator awake with trend) and bear markets (mean reversion when sleeping)

name = "12h_Williams_Alligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 2.0x 20-period average (20*12h = 10 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_1d = smma(close_1d, 13)
    teeth_1d = smma(close_1d, 8)
    lips_1d = smma(close_1d, 5)
    
    # Shift the lines as per Alligator definition
    jaw_1d_shifted = np.roll(jaw_1d, 8)
    teeth_1d_shifted = np.roll(teeth_1d, 5)
    lips_1d_shifted = np.roll(lips_1d, 3)
    
    # Set NaN for shifted positions
    jaw_1d_shifted[:8] = np.nan
    teeth_1d_shifted[:5] = np.nan
    lips_1d_shifted[:3] = np.nan
    
    # Align to 12h timeframe (wait for daily close)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d_shifted)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d_shifted)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator and indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator sleeping condition: all lines intertwined (market ranging)
        # Alligator awake condition: lines are separated and ordered (market trending)
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        
        # Check if Alligator is sleeping (lines intertwined)
        sleeping = (abs(jaw - teeth) < (close[i] * 0.001) and 
                   abs(teeth - lips) < (close[i] * 0.001) and
                   abs(lips - jaw) < (close[i] * 0.001))
        
        # Check if Alligator is awake and bullish (lips > teeth > jaw)
        bullish = lips > teeth > jaw
        
        # Check if Alligator is awake and bearish (jaw > teeth > lips)
        bearish = jaw > teeth > lips
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator awake AND bullish AND price > 1d EMA50 AND volume spike
            if (not sleeping and bullish and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator awake AND bearish AND price < 1d EMA50 AND volume spike
            elif (not sleeping and bearish and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator starts sleeping OR price below 1d EMA50 (trend change)
            if sleeping or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator starts sleeping OR price above 1d EMA50 (trend change)
            if sleeping or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals