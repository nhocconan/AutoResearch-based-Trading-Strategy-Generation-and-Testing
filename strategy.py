#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination
# ADX > 25 indicates trending market
# Alligator: Jaw (13), Teeth (8), Lips (5) SMAs with forward shift
# Long when Lips > Teeth > Jaw (bullish alignment) and ADX > 25
# Short when Lips < Teeth < Jaw (bearish alignment) and ADX > 25
# Exit when Alligator lines cross or ADX drops below 20
# Designed for 12-37 trades/year on 6h timeframe with strong trend following and low turnover

name = "6h_1d_adx_alligator_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smoothed_avg(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values are smoothed
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            else:
                result[i] = np.nan
        return result
    
    period = 14
    tr_smooth = smoothed_avg(tr, period)
    dm_plus_smooth = smoothed_avg(dm_plus, period)
    dm_minus_smooth = smoothed_avg(dm_minus, period)
    
    # DI values
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_avg(dx, period)
    adx_1d = adx
    
    # Calculate Alligator SMAs from 1d data
    jaw_period, teeth_period, lips_period = 13, 8, 5
    jaw_shift, teeth_shift, lips_shift = 8, 5, 3
    
    jaw = pd.Series(close_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(close_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(close_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Apply forward shift (future values become available only after shift periods)
    jaw = np.concatenate([np.full(jaw_shift, np.nan), jaw[:-jaw_shift]]) if len(jaw) > jaw_shift else np.full_like(jaw, np.nan)
    teeth = np.concatenate([np.full(teeth_shift, np.nan), teeth[:-teeth_shift]]) if len(teeth) > teeth_shift else np.full_like(teeth, np.nan)
    lips = np.concatenate([np.full(lips_shift, np.nan), lips[:-lips_shift]]) if len(lips) > lips_shift else np.full_like(lips, np.nan)
    
    # Align to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # ADX filter: trend strength
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] < 20  # exit threshold
        
        # Alligator alignment
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Entry conditions
        long_entry = bullish_alignment and strong_trend
        short_entry = bearish_alignment and strong_trend
        
        # Exit conditions
        long_exit = weak_trend or (lips_aligned[i] < jaw_aligned[i])  # lips cross below jaw or weak trend
        short_exit = weak_trend or (lips_aligned[i] > jaw_aligned[i])  # lips cross above jaw or weak trend
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals