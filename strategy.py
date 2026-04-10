#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + ADX trend strength + volume confirmation
# - Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price
# - Trend condition: Alligator lines aligned (Lips > Teeth > Jaw for long, reverse for short)
# - ADX > 25 confirms strong trend (avoids choppy markets)
# - Volume > 1.5x 20-period average confirms breakout momentum
# - Exit when Alligator lines cross (trend weakness) or volume drops below average
# - Designed for 6h timeframe to capture medium-term trends in both bull and bear markets
# - Target: 12-35 trades/year (50-140 total over 4 years) to minimize fee drag

name = "6h_1d_alligator_adx_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Williams Alligator
    # Median price = (high + low) / 2
    median_price = (df_1d['high'] + df_1d['low']) / 2
    close_1d = df_1d['close'].values
    median_price_values = median_price.values
    
    # Jaw: 13-period SMMA, 8 bars ahead
    jaw = pd.Series(median_price_values).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, 5 bars ahead
    teeth = pd.Series(median_price_values).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, 3 bars ahead
    lips = pd.Series(median_price_values).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Pre-compute 1d ADX (14-period)
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
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    vol_normal = prices['volume'] < volume_20_avg  # Exit when volume drops below average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get Alligator values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        
        # Check Alligator alignment
        lips_above_teeth = lips_val > teeth_val
        teeth_above_jaw = teeth_val > jaw_val
        lips_below_teeth = lips_val < teeth_val
        teeth_below_jaw = teeth_val < jaw_val
        
        if position == 0:  # Flat - look for new entries
            # Long condition: Lips > Teeth > Jaw (bullish alignment) AND ADX > 25 AND volume spike
            if (lips_above_teeth and teeth_above_jaw and adx_val > 25 and vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short condition: Lips < Teeth < Jaw (bearish alignment) AND ADX > 25 AND volume spike
            elif (lips_below_teeth and teeth_below_jaw and adx_val > 25 and vol_spike[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator lines cross (trend weakness)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (lips_val <= teeth_val or teeth_val <= jaw_val or vol_normal[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (lips_val >= teeth_val or teeth_val >= jaw_val or vol_normal[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals