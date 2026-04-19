#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d volume confirmation and ADX trend filter
# - Williams Alligator: Jaw (13-period SMA shifted 8), Teeth (8-period SMA shifted 5), Lips (5-period SMA shifted 3)
# - Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment)
# - Entry only when 1d volume > 1.5x 20-period average for conviction
# - Exit when Alligator lines cross in opposite direction or volume drops below average
# - Designed to capture trends in both bull and bear markets with clear entry/exit rules
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "4h_WilliamsAlligator_1dVolume_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator
    df_4h = get_htf_data(prices, '4h')
    
    # Williams Alligator components
    close_4h = df_4h['close'].values
    jaw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # Shifted 8 bars forward
    teeth = pd.Series(close_4h).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # Shifted 5 bars forward
    lips = pd.Series(close_4h).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # Shifted 3 bars forward
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips_vals)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1d data for ADX trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.absolute(high_1d[1:] - close_1d[:-1]),
        np.absolute(low_1d[1:] - close_1d[:-1])
    )
    # Add first element for alignment
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # ADX filter: trend strength > 25
        trend_filter = adx_aligned[i] > 25
        
        # Williams Alligator signals
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Look for long entry: bullish alignment + volume + trend
            if bullish_alignment and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: bearish alignment + volume + trend
            elif bearish_alignment and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on bearish alignment or weak trend
            if bearish_alignment or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on bullish alignment or weak trend
            if bullish_alignment or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals