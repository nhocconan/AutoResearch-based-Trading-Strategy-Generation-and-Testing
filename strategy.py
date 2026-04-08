#!/usr/bin/env python3
"""
4h Williams Alligator with 12h volume confirmation and 1d ADX trend filter
Hypothesis: The Williams Alligator (Jaw, Teeth, Lips) identifies trends when lines are aligned and separated.
Strong trends occur when Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish).
Combined with 12h volume expansion (>1.5x average) and 1d ADX > 25 for trend strength,
this captures sustained moves while avoiding whipsaw. Works in both bull and bear markets
by requiring strong daily trend alignment. Uses 4h timeframe as required.
"""

name = "4h_williams_alligator_12h_vol_1d_adx_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(close, jaw_period=13, teeth_period=8, lips_period=5,
                        jaw_shift=8, teeth_shift=5, lips_shift=3):
    """Calculate Williams Alligator lines: Jaw, Teeth, Lips"""
    # Smoothed median price (typical price)
    # Since we only have close, we'll use close as approximation
    # In practice, Alligator uses (high+low+close)/3 but we adapt for available data
    smoothed = pd.Series(close).rolling(window=2, min_periods=1).mean()
    
    jaw = pd.Series(smoothed).rolling(window=jaw_period, min_periods=jaw_period).mean()
    jaw = jaw.shift(jaw_shift)  # Shift forward by jaw_shift
    
    teeth = pd.Series(smoothed).rolling(window=teeth_period, min_periods=teeth_period).mean()
    teeth = teeth.shift(teeth_shift)  # Shift forward by teeth_shift
    
    lips = pd.Series(smoothed).rolling(window=lips_period, min_periods=lips_period).mean()
    lips = lips.shift(lips_shift)  # Shift forward by lips_shift
    
    return jaw.values, teeth.values, lips.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume average (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 4h close data
    jaw, teeth, lips = calculate_alligator(close)
    
    # Calculate 14-period ADX for 1d data
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_1d = np.concatenate([[0], dm_plus_1d])
    dm_minus_1d = np.concatenate([[0], dm_minus_1d])
    
    # Smoothed values with proper min_periods
    tr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_1d = pd.Series(dm_plus_1d).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_1d = pd.Series(dm_minus_1d).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_1d = 100 * dm_plus_14_1d / tr14_1d
    di_minus_1d = 100 * dm_minus_14_1d / tr14_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # 12-period volume average for confirmation on 12h data
    vol_avg_12 = pd.Series(volume_12h).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback (Alligator needs ~13+8 shifts)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d[i]) or 
            np.isnan(vol_avg_12[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d value for current 4h bar
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        # Get aligned 12h value for current 4h bar
        vol_avg_12_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12)[i]
        
        # Regime filter: only trade in strong trending markets on daily
        strong_trend_1d = adx_1d_aligned > 25
        
        # Volume confirmation: current volume > 1.5x 12-period average on 12h
        volume_confirm = volume[i] > 1.5 * vol_avg_12_aligned
        
        if position == 1:  # Long position
            # Exit: Alligator lines converge (Lips crosses below Teeth) OR weak trend
            if lips[i] < teeth[i] or not strong_trend_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines converge (Lips crosses above Teeth) OR weak trend
            if lips[i] > teeth[i] or not strong_trend_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation and in strong trending markets on daily
            if volume_confirm and strong_trend_1d:
                # Bullish alignment: Lips > Teeth > Jaw
                if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish alignment: Lips < Teeth < Jaw
                elif lips[i] < teeth[i] and teeth[i] < jaw[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals