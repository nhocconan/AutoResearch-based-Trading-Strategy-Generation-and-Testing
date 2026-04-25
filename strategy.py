#!/usr/bin/env python3
"""
1d Williams Alligator + Volume Spike + Choppiness Regime Filter
Hypothesis: Williams Alligator identifies trend direction (jaw/teeth/lips alignment).
Combined with volume confirmation and choppiness regime filter (CHOP < 50 = trending),
this strategy captures strong trending moves while avoiding choppy markets.
Uses 1w HTF for Alligator calculation to reduce noise and targets 30-100 trades over 4 years.
Works in both bull (long when lips>teeth>jaw) and bear (short when lips<teeth<jaw) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Alligator calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1w timeframe
    # Jaw (Blue): 13-period SMMA, shifted 8 bars
    # Teeth (Red): 8-period SMMA, shifted 5 bars  
    # Lips (Green): 5-period SMMA, shifted 3 bars
    close_1w = pd.Series(df_1w['close'])
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    jaw_1w = close_1w.ewm(alpha=1/13, adjust=False).mean().shift(8).values
    teeth_1w = close_1w.ewm(alpha=1/8, adjust=False).mean().shift(5).values
    lips_1w = close_1w.ewm(alpha=1/5, adjust=False).mean().shift(3).values
    
    # Align Alligator lines to 1d timeframe
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Choppiness Index on 1d timeframe (regime filter)
    # CHOP < 50 = trending market (good for trend following)
    # CHOP >= 50 = ranging/choppy market (avoid)
    atr_period = 14
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.concatenate([[close[0]], close[:-1]]))),
                    np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # True Range sum over period
    tr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    # Max(high) - Min(low) over period
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    range_max_min = max_high - min_low
    
    # Avoid division by zero
    chop_denominator = np.where(range_max_min == 0, 1, range_max_min)
    chop = 100 * np.log10(tr_sum / chop_denominator) / np.log10(atr_period)
    chop = np.where(np.isnan(chop), 50, chop)  # Default to 50 (neutral) if NaN
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator and CHOP calculations
    start_idx = max(50, 34)  # Alligator needs ~50 bars, CHOP needs 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or 
            np.isnan(lips_1w_aligned[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        curr_chop = chop[i]
        
        # Alligator alignment signals
        bullish_alignment = (lips_1w_aligned[i] > teeth_1w_aligned[i] > jaw_1w_aligned[i])
        bearish_alignment = (lips_1w_aligned[i] < teeth_1w_aligned[i] < jaw_1w_aligned[i])
        
        # Regime filter: only trade in trending markets (CHOP < 50)
        trending_market = curr_chop < 50
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + volume spike + trending market
            # Long: bullish alignment AND volume spike AND trending market
            long_entry = bullish_alignment and vol_spike and trending_market
            # Short: bearish alignment AND volume spike AND trending market
            short_entry = bearish_alignment and vol_spike and trending_market
            
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
            # Exit: bearish Alligator alignment OR loss of volume confirmation OR choppy market
            if (not bullish_alignment) or (not vol_spike) or (not trending_market):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: bullish Alligator alignment OR loss of volume confirmation OR choppy market
            if (not bearish_alignment) or (not vol_spike) or (not trending_market):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_VolumeSpike_ChopFilter"
timeframe = "1d"
leverage = 1.0