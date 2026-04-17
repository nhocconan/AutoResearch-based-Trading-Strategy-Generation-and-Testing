#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Alligator indicator (1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price_1w = (high_1w := (df_1w['high'].values + df_1w['low'].values) / 2) if hasattr(df_1w, 'high') else (df_1w['high'] + df_1w['low']) / 2
    # Actually compute properly:
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    median_price_1w = (high_1w + low_1w) / 2
    
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw_1w = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth_1w = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMMA, 3 bars ahead)
    lips_1w = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe
    jaw_12h = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_12h = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_12h = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Get daily data for volume confirmation and ATR
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily volume average (20-period)
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_12h = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or
            np.isnan(atr_12h[i]) or np.isnan(volume_ma20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current daily volume > 1.5 * 20-day average
        # Note: We use daily volume aligned to 12h, so we need to check if current 12h bar
        # falls within a day with high volume
        volume_filter = volume[i] > (1.5 * volume_ma20_12h[i])
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = lips_12h[i] > teeth_12h[i] and teeth_12h[i] > jaw_12h[i]
        bearish_alignment = lips_12h[i] < teeth_12h[i] and teeth_12h[i] < jaw_12h[i]
        
        if position == 0:
            # Long entry: bullish alignment + volume
            if bullish_alignment and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment + volume
            elif bearish_alignment and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment forms OR volatility drops too low
            if bearish_alignment or (atr_12h[i] < 0.5 * atr_12h[i-5] if i >= 5 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment forms OR volatility drops too low
            if bullish_alignment or (atr_12h[i] < 0.5 * atr_12h[i-5] if i >= 5 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_VolumeFilter"
timeframe = "12h"
leverage = 1.0