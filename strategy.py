#!/usr/bin/env python3
"""
4h_Williams_Alligator_Regime_Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
In bull/bear markets: trade in direction of aligned Alligator (all three lines ordered).
In ranging markets (Choppiness Index > 61.8): fade extremes at Bollinger Bands (20,2.0).
Uses 4h for execution, 1d for Alligator/Choppiness regime filter. Volume confirmation required.
Target: 20-50 trades/year per symbol (80-200 total over 4 years). Works in bull (trend follow) and bear (mean revert in chop).
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
    
    # Get daily data for regime and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1d: SMA of median price (HL/2)
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars  
    # Lips: 5-period SMA, shifted 3 bars
    hl2_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw = pd.Series(hl2_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(hl2_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(hl2_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 4h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Choppiness Index on 1d (regime filter)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    # High CHOP (>61.8) = ranging, Low CHOP (<38.2) = trending
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr14 / 14) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Bollinger Bands on 1d for mean reversion signals in chop
    bb_mid = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + (bb_std * 2.0)
    bb_lower = bb_mid - (bb_std * 2.0)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Volume confirmation on 4h: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for Alligator + Chop + BB)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime determination
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        # Transition zone (38.2-61.8) = no new entries, hold existing
        
        if position == 0:
            if is_trending:
                # Trend following: Alligator aligned (Lips > Teeth > Jaw = uptrend, reverse for downtrend)
                if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]:
                    # Uptrend: long on pullback to Teeth with volume
                    if close[i] <= teeth_aligned[i] * 1.005 and volume_spike[i]:  # within 0.5% of Teeth
                        signals[i] = 0.25
                        position = 1
                elif lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]:
                    # Downtrend: short on pullback to Teeth with volume
                    if close[i] >= teeth_aligned[i] * 0.995 and volume_spike[i]:  # within 0.5% of Teeth
                        signals[i] = -0.25
                        position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Mean reversion: fade Bollinger Band extremes
                if close[i] <= bb_lower_aligned[i] and volume_spike[i]:
                    # Oversold: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= bb_upper_aligned[i] and volume_spike[i]:
                    # Overbought: short
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition zone: no new entries
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            if is_trending:
                # Exit trend long if Alligator reverses or price closes below Jaw
                if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or close[i] < jaw_aligned[i]:
                    signals[i] = 0.0
                    position = 0
            elif is_ranging:
                # Exit range long at midpoint or upper band
                if close[i] >= bb_mid[len(bb_mid)-len(bb_mid)+i] if hasattr(bb_mid, '__getitem__') else bb_mid[i] if i < len(bb_mid) else bb_mid[-1]:
                    # Simplified: exit at BB middle
                    signals[i] = 0.0
                    position = 0
            else:
                # Transition zone: exit
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            if is_trending:
                # Exit trend short if Alligator reverses or price closes above Jaw
                if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or close[i] > jaw_aligned[i]:
                    signals[i] = 0.0
                    position = 0
            elif is_ranging:
                # Exit range short at midpoint or lower band
                if close[i] <= bb_mid[len(bb_mid)-len(bb_mid)+i] if hasattr(bb_mid, '__getitem__') else bb_mid[i] if i < len(bb_mid) else bb_mid[-1]:
                    signals[i] = 0.0
                    position = 0
            else:
                # Transition zone: exit
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Williams_Alligator_Regime_Filter"
timeframe = "4h"
leverage = 1.0