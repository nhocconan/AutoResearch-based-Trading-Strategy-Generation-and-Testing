#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with volume confirmation and ADX filter
# Williams Alligator uses SMAs (5,8,13) to identify trend direction
# Jaw (13-period), Teeth (8-period), Lips (5-period) - when aligned, trend is strong
# In uptrend: Lips > Teeth > Jaw; Downtrend: Lips < Teeth < Jaw
# Volume > 1.5x average confirms momentum
# ADX > 20 ensures trending market (avoids choppy sideways)
# Target: 25-40 trades/year per symbol to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator SMAs on 1d
    jaw_period = 13   # Blue line
    teeth_period = 8  # Red line  
    lips_period = 5   # Green line
    
    close_1d = df_1d['close'].values
    
    # Jaw (13-period SMA)
    jaw = pd.Series(close_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    # Teeth (8-period SMA) 
    teeth = pd.Series(close_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    # Lips (5-period SMA)
    lips = pd.Series(close_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Load 1h data ONCE for ADX
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 1h ADX (14 periods)
    adx_len = 14
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1 = high_1h[1:] - low_1h[1:]
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1h[1:] - high_1h[:-1]) > (low_1h[:-1] - low_1h[1:]), 
                       np.maximum(high_1h[1:] - high_1h[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1h[:-1] - low_1h[1:]) > (high_1h[1:] - high_1h[:-1]), 
                        np.maximum(low_1h[:-1] - low_1h[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1h, adx)
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, jaw_period, teeth_period, lips_period, adx_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_aligned[i] > 20
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Williams Alligator alignment signals
        # Uptrend: Lips > Teeth > Jaw
        uptrend_aligned = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Downtrend: Lips < Teeth < Jaw
        downtrend_aligned = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Enter long: Alligator aligned up + volume + trend
            if (uptrend_aligned and 
                volume_confirmed and 
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: Alligator aligned down + volume + trend
            elif (downtrend_aligned and 
                  volume_confirmed and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips < Teeth or Teeth < Jaw)
            if not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips > Teeth or Teeth > Jaw)
            if not (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Williams_Alligator_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0