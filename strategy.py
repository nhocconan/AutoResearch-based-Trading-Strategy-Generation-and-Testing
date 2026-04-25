#!/usr/bin/env python3
"""
4h Williams Alligator + 1d EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Williams Alligator identifies trend alignment (jaws/teeth/lips); 1d EMA50 ensures alignment with daily trend; volume spike confirms conviction; chop filter avoids whipsaws in ranging markets. Designed for 4h timeframe to target 20-50 trades/year (80-200 over 4 years), minimizing fee drag. Works in both bull and bear markets by following the daily trend and avoiding counter-trend entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1d: SMAs with offsets
    # Jaw: 13-period SMMA smoothed 8 bars ahead
    # Teeth: 8-period SMMA smoothed 5 bars ahead  
    # Lips: 5-period SMMA smoothed 3 bars ahead
    close_1d = pd.Series(df_1d['close'])
    # Smoothed Moving Average (SMMA) approximation using EMA with alpha=1/period
    jaw_raw = close_1d.ewm(alpha=1/13, adjust=False).mean()
    teeth_raw = close_1d.ewm(alpha=1/8, adjust=False).mean()
    lips_raw = close_1d.ewm(alpha=1/5, adjust=False).mean()
    # Apply offsets (shift forward)
    jaw = jaw_raw.shift(8).values  # 8 bars ahead
    teeth = teeth_raw.shift(5).values  # 5 bars ahead
    lips = lips_raw.shift(3).values  # 3 bars ahead
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d EMA50 for trend filter
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Chop filter: avoid trading in choppy markets (Choppiness Index > 61.8)
    # Calculate ATR and its rolling sum for Chop
    tr1 = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum_atr / (hh - ll)) / log10(14)
    chop = 100 * np.log10(sum_atr / np.maximum(hh - ll, 1e-10)) / np.log10(14)
    chop_filter = chop < 61.8  # Only trade when NOT choppy (trending market)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(50, 20, 20, 14, 14)  # EMA50, volume MA, Chop components
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        chop_ok = chop_filter[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish; Lips < Teeth < Jaw = bearish
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_50_aligned[i]
        bearish_bias = curr_close < ema_50_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + trend + volume + chop filter
            # Long: bullish Alligator alignment AND bullish bias AND volume spike AND chop filter
            long_entry = bullish_alignment and bullish_bias and vol_spike and chop_ok
            # Short: bearish Alligator alignment AND bearish bias AND volume spike AND chop filter
            short_entry = bearish_alignment and bearish_bias and vol_spike and chop_ok
            
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
            # Exit: Alligator turns bearish OR loss of bullish bias OR chop becomes too high
            if (not bullish_alignment) or (curr_close < ema_50_aligned[i]) or (not chop_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator turns bullish OR loss of bearish bias OR chop becomes too high
            if (not bearish_alignment) or (curr_close > ema_50_aligned[i]) or (not chop_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0