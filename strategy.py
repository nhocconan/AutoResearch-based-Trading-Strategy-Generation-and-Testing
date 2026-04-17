#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator components on daily
    jaw_1d = pd.Series(close_1d).rolling(window=13, center=False, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(close_1d).rolling(window=8, center=False, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(close_1d).rolling(window=5, center=False, min_periods=5).mean().shift(3).values
    
    # Align Alligator to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # ADX on 6h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = (high_series - close_series.shift()).abs()
    tr3 = (low_series - close_series.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_dm = high_series.diff()
    minus_dm = low_series.diff().multiply(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    smoothed_plus_dm = plus_dm.rolling(window=14, min_periods=14).mean()
    smoothed_minus_dm = minus_dm.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * smoothed_plus_dm / atr
    minus_di = 100 * smoothed_minus_dm / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Volume filter: current > 1.5x 20-period avg
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw and ADX > 25
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) and adx[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Bearish alignment: Lips < Teeth < Jaw and ADX > 25
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) and adx[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: alignment breaks or ADX drops
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: alignment breaks or ADX drops
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ADX_VolumeFilter"
timeframe = "6h"
leverage = 1.0