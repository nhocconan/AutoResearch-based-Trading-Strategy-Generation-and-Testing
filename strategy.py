#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Volume Confirmation
# Uses Williams Alligator (Jaw: SMA13, Teeth: SMA8, Lips: SMA5) on 12h timeframe
# Long when Lips > Teeth > Jaw (bullish alignment) with 1d volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) with 1d volume spike
# Exits when alignment breaks or volume drops below average
# Williams Alligator identifies trend phases; volume confirms strength; avoids whipsaws in ranging markets
# Target: 30-80 trades per symbol over 4 years (7.5-20/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    median_price_12h = (high_12h + low_12h) / 2
    
    # Jaw (blue line): 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth (red line): 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips (green line): 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Calculate 1d ADX for trend strength
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
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_vals)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Alligator and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish = (lips_aligned[i] > teeth_aligned[i] and 
                      teeth_aligned[i] > jaw_aligned[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = (lips_aligned[i] < teeth_aligned[i] and 
                      teeth_aligned[i] < jaw_aligned[i])
            
            # Long setup: bullish alignment with volume spike and trend
            if (bullish and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                adx_aligned[i] > 25):                           # Strong trend
                position = 1
                signals[i] = position_size
            # Short setup: bearish alignment with volume spike and trend
            elif (bearish and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                  adx_aligned[i] > 25):                           # Strong trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: alignment breaks or volume drops
            bullish = (lips_aligned[i] > teeth_aligned[i] and 
                      teeth_aligned[i] > jaw_aligned[i])
            if not bullish or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: alignment breaks or volume drops
            bearish = (lips_aligned[i] < teeth_aligned[i] and 
                      teeth_aligned[i] < jaw_aligned[i])
            if not bearish or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Williams_Alligator_1dVolume_ADX"
timeframe = "12h"
leverage = 1.0