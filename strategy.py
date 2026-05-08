#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d ADX filter and volume confirmation
# Uses Alligator (Jaw/Teeth/Lips) from 6h for trend direction, 1d ADX>25 to filter strong trends,
# and volume > 1.5x 20-period average for confirmation. Avoids whipsaws in weak trends.
# Targets 50-150 total trades over 4 years (12-37/year) with disciplined entries.

name = "6h_WilliamsAlligator_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams Alligator (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator lines on 6h median price
    median_price = (df_6h['high'].values + df_6h['low'].values) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d high/low/close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.append([close_1d[0]], close_1d[:-1]))
    tr3 = np.abs(low_1d - np.append([close_1d[0]], close_1d[:-1]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.append([high_1d[0]], high_1d[:-1])) > 
                       (np.append([low_1d[0]], low_1d[:-1]) - low_1d), 
                       np.maximum(high_1d - np.append([high_1d[0]], high_1d[:-1]), 0), 0)
    dm_minus = np.where((np.append([low_1d[0]], low_1d[:-1]) - low_1d) > 
                        (high_1d - np.append([high_1d[0]], high_1d[:-1])), 
                        np.maximum(np.append([low_1d[0]], low_1d[:-1]) - low_1d, 0), 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_sum = wilders_smoothing(tr, 14)
    dm_plus_sum = wilders_smoothing(dm_plus, 14)
    dm_minus_sum = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_sum > 0, 100 * dm_plus_sum / tr_sum, 0)
    di_minus = np.where(tr_sum > 0, 100 * dm_minus_sum / tr_sum, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for ADX and Alligator
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        if position == 0:
            # Enter long: Lips > Teeth > Jaw, ADX > 25, volume spike
            if lips_val > teeth_val > jaw_val and adx_val > 25 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw, ADX > 25, volume spike
            elif lips_val < teeth_val < jaw_val and adx_val > 25 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator weakens (Lips < Teeth) or ADX < 20
            if lips_val < teeth_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator weakens (Lips > Teeth) or ADX < 20
            if lips_val > teeth_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals