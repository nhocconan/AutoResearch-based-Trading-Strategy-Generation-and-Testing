#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Choppiness regime filter + volume confirmation
# Williams Alligator identifies trend via three SMAs (Jaw=13, Teeth=8, Lips=5)
# 1d Choppiness Index > 61.8 = ranging (fade extremes), < 38.2 = trending (follow Alligator)
# Volume confirmation (1.5x 20-period average) ensures participation
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by adapting to regime via Choppiness Index
# Uses 1d for HTF regime and Alligator calculation for stability

name = "12h_WilliamsAlligator_1dChopRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Choppiness regime and Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator: three SMAs
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = pd.Series(df_1d['close']).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(df_1d['close']).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(df_1d['close']).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Calculate 1d Choppiness Index
    chop_period = 14
    # True Range
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of True Range over chop_period
    tr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Highest high and lowest low over chop_period
    hh = pd.Series(df_1d['high'].values).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(df_1d['low'].values).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Choppiness Index formula
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(chop_period)
    chop = np.where((hh - ll) == 0, 50, chop)  # avoid division by zero
    
    # Align 1d indicators to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator, Choppiness and volume MA)
    start_idx = 50  # max(20 for volume, 34 for Alligator/Choppiness) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1d Choppiness Index
        trending = chop_aligned[i] < 38.2
        ranging = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # In trending market: follow Alligator alignment (Lips > Teeth > Jaw = bullish)
                # Long: Bullish alignment AND previous not bullish (momentum shift up)
                bullish_aligned = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
                bullish_prev = lips_aligned[i-1] > teeth_aligned[i-1] and teeth_aligned[i-1] > jaw_aligned[i-1]
                if bullish_aligned and not bullish_prev and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bearish alignment (Lips < Teeth < Jaw) AND previous not bearish
                bearish_aligned = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
                bearish_prev = lips_aligned[i-1] < teeth_aligned[i-1] and teeth_aligned[i-1] < jaw_aligned[i-1]
                if bearish_aligned and not bearish_prev and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # ranging or transition regime
                # In ranging market: fade Alligator extremes
                # Long: Lips significantly below Jaw (oversold) AND previous not oversold
                lips_below_jaw = lips_aligned[i] < jaw_aligned[i]
                lips_below_jaw_prev = lips_aligned[i-1] < jaw_aligned[i-1]
                if lips_below_jaw and not lips_below_jaw_prev and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Lips significantly above Jaw (overbought) AND previous not overbought
                lips_above_jaw = lips_aligned[i] > jaw_aligned[i]
                lips_above_jaw_prev = lips_aligned[i-1] > jaw_aligned[i-1]
                if lips_above_jaw and not lips_above_jaw_prev and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending long when Alligator turns bearish
                bearish_aligned = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
                if bearish_aligned:
                    exit_signal = True
            else:
                # Exit ranging long when Lips rises back above Jaw (weakening oversold)
                if lips_aligned[i] >= jaw_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending short when Alligator turns bullish
                bullish_aligned = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
                if bullish_aligned:
                    exit_signal = True
            else:
                # Exit ranging short when Lips falls back below Jaw (weakening overbought)
                if lips_aligned[i] <= jaw_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals