#!/usr/bin/env python3
"""
6h_WilliamsAlligator_DecisionPoint_v1
Hypothesis: Use Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) on 6h for trend direction, with decision point on Teeth crossing Jaw/Lips. 
Add 1d ADX > 25 filter to confirm trending regime. Enter on cross with 1d volume spike > 1.5x average. 
Exit when price closes outside Alligator mouth (below Lips for long, above Teeth for short) or ADX < 20.
Target: 15-25 trades/year by requiring strong trend confirmation. Works in bull/bear via directional Alligator signals.
"""

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
    
    # Get 6h data for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMAs
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw_6h = np.full_like(close_6h, np.nan)
    teeth_6h = np.full_like(close_6h, np.nan)
    lips_6h = np.full_like(close_6h, np.nan)
    
    # Calculate SMAs
    if len(close_6h) >= jaw_period:
        for i in range(jaw_period, len(close_6h)):
            jaw_6h[i] = np.mean(close_6h[i-jaw_period:i])
    
    if len(close_6h) >= teeth_period:
        for i in range(teeth_period, len(close_6h)):
            teeth_6h[i] = np.mean(close_6h[i-teeth_period:i])
    
    if len(close_6h) >= lips_period:
        for i in range(lips_period, len(close_6h)):
            lips_6h[i] = np.mean(close_6h[i-lips_period:i])
    
    # Align Alligator lines to 6h timeframe
    jaw_6h_aligned = align_htf_to_ltf(prices, df_6h, jaw_6h)
    teeth_6h_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h)
    lips_6h_aligned = align_htf_to_ltf(prices, df_6h, lips_6h)
    
    # Get 1d data for ADX and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ADX(14)
    adx_period = 14
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with index
    
    dm_plus_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_1d = np.concatenate([[np.nan], dm_plus_1d])
    dm_minus_1d = np.concatenate([[np.nan], dm_minus_1d])
    
    # Smoothed TR, DM+ and DM-
    atr_1d = np.full_like(close_1d, np.nan)
    dm_plus_smoothed = np.full_like(close_1d, np.nan)
    dm_minus_smoothed = np.full_like(close_1d, np.nan)
    
    if len(tr_1d) >= adx_period:
        # Initial values
        atr_1d[adx_period] = np.nanmean(tr_1d[1:adx_period+1])
        dm_plus_smoothed[adx_period] = np.nanmean(dm_plus_1d[1:adx_period+1])
        dm_minus_smoothed[adx_period] = np.nanmean(dm_minus_1d[1:adx_period+1])
        
        # Wilder smoothing
        for i in range(adx_period + 1, len(close_1d)):
            atr_1d[i] = (atr_1d[i-1] * (adx_period - 1) + tr_1d[i]) / adx_period
            dm_plus_smoothed[i] = (dm_plus_smoothed[i-1] * (adx_period - 1) + dm_plus_1d[i]) / adx_period
            dm_minus_smoothed[i] = (dm_minus_smoothed[i-1] * (adx_period - 1) + dm_minus_1d[i]) / adx_period
    
    # DI+ and DI-
    di_plus_1d = np.full_like(close_1d, np.nan)
    di_minus_1d = np.full_like(close_1d, np.nan)
    dx_1d = np.full_like(close_1d, np.nan)
    
    valid = ~np.isnan(atr_1d) & (atr_1d != 0)
    di_plus_1d[valid] = 100 * dm_plus_smoothed[valid] / atr_1d[valid]
    di_minus_1d[valid] = 100 * dm_minus_smoothed[valid] / atr_1d[valid]
    
    # DX and ADX
    dx_denom = di_plus_1d + di_minus_1d
    dx_1d[valid & (dx_denom != 0)] = 100 * np.abs(di_plus_1d[valid] - di_minus_1d[valid]) / dx_denom[valid & (dx_denom != 0)]
    
    adx_1d = np.full_like(close_1d, np.nan)
    if len(dx_1d) >= adx_period:
        # Initial ADX
        valid_dx = ~np.isnan(dx_1d)
        if np.sum(valid_dx) >= adx_period:
            adx_1d[2*adx_period-1] = np.nanmean(dx_1d[adx_period:2*adx_period])
            
            # Wilder smoothing for ADX
            for i in range(2*adx_period, len(close_1d)):
                if not np.isnan(dx_1d[i]) and not np.isnan(adx_1d[i-1]):
                    adx_1d[i] = (adx_1d[i-1] * (adx_period - 1) + dx_1d[i]) / adx_period
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d volume average
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    vol_period = 20
    
    if len(volume_1d) >= vol_period:
        for i in range(vol_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i-vol_period:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period, adx_period, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_6h_aligned[i]) or np.isnan(teeth_6h_aligned[i]) or 
            np.isnan(lips_6h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check for Alligator alignment (trending condition)
        # Bullish: Lips > Teeth > Jaw (green alignment)
        bullish_align = lips_6h_aligned[i] > teeth_6h_aligned[i] and teeth_6h_aligned[i] > jaw_6h_aligned[i]
        # Bearish: Lips < Teeth < Jaw (red alignment)
        bearish_align = lips_6h_aligned[i] < teeth_6h_aligned[i] and teeth_6h_aligned[i] < jaw_6h_aligned[i]
        
        # Volume confirmation: 1d volume spike
        vol_confirm = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # ADX trend strength filter
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Enter long: bullish alignment + strong trend + volume
            if bullish_align and strong_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + strong trend + volume
            elif bearish_align and strong_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment OR weak trend OR price outside mouth (below Lips)
            if bearish_align or weak_trend or close[i] < lips_6h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment OR weak trend OR price outside mouth (above Teeth)
            if bullish_align or weak_trend or close[i] > teeth_6h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_DecisionPoint_v1"
timeframe = "6h"
leverage = 1.0