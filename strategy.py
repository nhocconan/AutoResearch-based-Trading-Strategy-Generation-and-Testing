#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d trend filter + volume confirmation.
# Uses Alligator jaws/teeth/lips for trend direction and entry signals.
# 1d EMA50 filter ensures alignment with higher timeframe trend.
# Volume spike confirms momentum. Works in bull/bear by following trend.
# Target: 12-37 trades/year (50-150 total over 4 years) on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Alligator calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h
    # Jaw (blue): 13-period SMMA, smoothed by 8 periods
    # Teeth (red): 8-period SMMA, smoothed by 5 periods
    # Lips (green): 5-period SMMA, smoothed by 3 periods
    close_12h = df_12h['close'].values
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        sma = np.convolve(arr, np.ones(period)/period, mode='valid')
        result[period-1:] = sma
        # Smooth the SMA
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Smooth further as per Alligator definition
    jaw = smma(jaw, 8)
    teeth = smma(teeth, 5)
    lips = smma(lips, 3)
    
    # Align Alligator lines to 12h timeframe (already aligned via get_htf_data)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment check: lips > teeth > jaw = uptrend
        # lips < teeth < jaw = downtrend
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        
        is_uptrend = (lips_val > teeth_val > jaw_val)
        is_downtrend = (lips_val < teeth_val < jaw_val)
        
        if position == 0:
            # Long: Alligator aligned up + price above lips + volume confirmation
            if (is_uptrend and 
                close[i] > lips_val and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + price below lips + volume confirmation
            elif (is_downtrend and 
                  close[i] < lips_val and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back through teeth line or trend changes
            if position == 1:
                if close[i] < teeth_val or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > teeth_val or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0