#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d volume confirmation and ADX trend filter
# Uses Bill Williams Alligator (Jaw/Teeth/Lips) to identify trend direction
# Long when Lips > Teeth > Jaw (bullish alignment) + price above Teeth + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + price below Teeth + volume spike
# ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions
# Designed for low trade frequency (target 15-30/year) with clear trend following logic
# Works in both bull (catching strong uptrends) and bear (catching strong downtrends) markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Alligator calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Williams Alligator (13,8,5) smoothed with 8,5,3 periods respectively
    # Jaw (Blue) - 13-period SMMA smoothed 8 periods
    jaw_raw = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth (Red) - 8-period SMMA smoothed 5 periods
    teeth_raw = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips (Green) - 5-period SMMA smoothed 3 periods
    lips_raw = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ADX for trend filter (14-period on 6h)
    # Calculate True Range
    tr1 = np.maximum(high_6h[1:], low_6h[:-1]) - np.minimum(high_6h[1:], low_6h[:-1])
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high_6h[1:] - high_6h[:-1]) > (low_6h[:-1] - low_6h[1:]), 
                       np.maximum(high_6h[1:] - high_6h[:-1], 0), 0)
    dm_minus = np.where((low_6h[:-1] - low_6h[1:]) > (high_6h[1:] - high_6h[:-1]), 
                        np.maximum(low_6h[:-1] - low_6h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (similar to EMA but different factor)
    tr_period = 14
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    # Initial values
    atr[tr_period] = np.nansum(tr[1:tr_period+1])
    dm_plus_smooth[tr_period] = np.nansum(dm_plus[1:tr_period+1])
    dm_minus_smooth[tr_period] = np.nansum(dm_minus[1:tr_period+1])
    
    # Wilder smoothing
    for i in range(tr_period + 1, len(tr)):
        atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period - 1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # Calculate DI+ and DI-
    di_plus = np.full_like(atr, np.nan)
    di_minus = np.full_like(atr, np.nan)
    dx = np.full_like(atr, np.nan)
    
    for i in range(tr_period, len(atr)):
        if atr[i] != 0:
            di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    # ADX is smoothed DX
    adx = np.full_like(dx, np.nan)
    adx[2*tr_period-1] = np.nansum(dx[tr_period:2*tr_period]) / tr_period
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    # Align all indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    adx_aligned = align_htf_to_ltf(prices, df_6h, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            continue
        
        # Only trade when ADX > 25 (trending market)
        if adx_aligned[i] <= 25:
            # Close position if we're in a ranging market
            if position != 0:
                position = 0
                signals[i] = 0.0
            continue
        
        # Bullish Alligator alignment: Lips > Teeth > Jaw
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        # Bearish Alligator alignment: Lips < Teeth < Jaw
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        # Long entry: Bullish alignment + price above Teeth + volume spike
        if (bullish_alignment and 
            close[i] > teeth_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bearish alignment + price below Teeth + volume spike
        elif (bearish_alignment and 
              close[i] < teeth_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Opposite Alligator alignment or price crosses Jaw
        elif position == 1 and (not bullish_alignment or close[i] < jaw_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_alignment or close[i] > jaw_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1dVolume_ADXFilter"
timeframe = "6h"
leverage = 1.0