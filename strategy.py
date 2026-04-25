#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + Chop Regime Filter
Hypothesis: Williams Alligator (Jaw=TEETH=LIPS SMMA) identifies trend absence (all lines intertwined = chop) 
vs trend presence (lines diverged = trend). In chop (Alligator sleeping), fade extremes via Donchian breakout failure.
In trend (Alligator awakened), trade breakouts in direction of alignment. Volume spike confirms participation.
12h timeframe targets 12-37 trades/year (50-150 over 4 years). Works in bull/bear via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) aka Wilder's MA"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    res = np.full_like(arr, np.nan, dtype=float)
    res[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        res[i] = (res[i-1] * (period-1) + arr[i]) / period
    return res

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMMA of median price
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)  # SMMA(13,8)
    teeth = smma(median_price, 8)  # SMMA(8,5)
    lips = smma(median_price, 5)   # SMMA(5,3)
    
    # Chop regime: Alligator sleeping (lines intertwined)
    # Measure: max distance between lines as % of ATR
    atr_raw = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    atr = align_htf_to_ltf(prices, df_1d, atr_raw)  # 1d ATR aligned to 12h
    jaw_teeth = np.abs(jaw - teeth)
    teeth_lips = np.abs(teeth - lips)
    lips_jaw = np.abs(lips - jaw)
    max_jaw_spread = np.maximum(jaw_teeth, np.maximum(teeth_lips, lips_jaw))
    chop_filter = max_jaw_spread < (0.5 * atr)  # Alligator sleeping = chop
    
    # Donchian channels (20-period) for breakouts
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 13, 34)  # Donchian, Alligator, 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        is_chop = chop_filter[i]
        
        if position == 0:
            # Look for entry signals
            if is_chop:
                # In chop: fade Donchian breakouts (mean reversion)
                # Long: price rejects lower Donchian (closes above low after touching/under)
                # Short: price rejects upper Donchian (closes below high after touching/over)
                long_entry = (curr_low <= donchian_low[i]) and (curr_close > donchian_low[i]) and vol_spike
                short_entry = (curr_high >= donchian_high[i]) and (curr_close < donchian_high[i]) and vol_spike
            else:
                # In trend: trade Donchian breakouts in direction of Alligator alignment
                # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
                lips_above_teeth = lips[i] > teeth[i]
                teeth_above_jaw = teeth[i] > jaw[i]
                uptrend_align = lips_above_teeth and teeth_above_jaw
                
                lips_below_teeth = lips[i] < teeth[i]
                teeth_below_jaw = teeth[i] < jaw[i]
                downtrend_align = lips_below_teeth and teeth_below_jaw
                
                # Long: price breaks above upper Donchian in uptrend alignment
                long_entry = (curr_high > donchian_high[i]) and uptrend_align and vol_spike
                # Short: price breaks below lower Donchian in downtrend alignment
                short_entry = (curr_low < donchian_low[i]) and downtrend_align and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator turns against trend OR price retouches opposite Donchian
            lips_below_teeth = lips[i] < teeth[i]
            teeth_below_jaw = teeth[i] < jaw[i]
            trend_against = lips_below_teeth and teeth_below_jaw  # Alligator turning down
            
            if trend_against or (curr_low <= donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator turns against trend OR price retouches opposite Donchian
            lips_above_teeth = lips[i] > teeth[i]
            teeth_above_jaw = teeth[i] > jaw[i]
            trend_against = lips_above_teeth and teeth_above_jaw  # Alligator turning up
            
            if trend_against or (curr_high >= donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_VolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0