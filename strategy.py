#!/usr/bin/env python3
"""
1d_HTF_Alligator_Trend_Regime_VolumeFilter
Hypothesis: On 1d timeframe, use Williams Alligator (jaw=13, teeth=8, lips=5) from 1w HTF to define regime: 
- Bull: Alligator aligned (lips > teeth > jaw) and price above lips
- Bear: Alligator aligned (jaw > teeth > lips) and price below jaw
- Chop: Alligator intertwined (no clear alignment)
Enter long only in bull regime on close > lips with volume > 1.5x 20-day average. 
Enter short only in bear regime on close < jaw with volume spike. 
Exit on regime change to chop or opposite regime. 
This captures strong trends while avoiding whipsaws in ranging markets. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Alligator (SMMA-based)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Williams Alligator: 3 smoothed moving averages (SMMA)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    close_1w = pd.Series(df_1w['close'].values)
    jaw_1w = close_1w.ewm(alpha=1/13, adjust=False).mean().values  # SMMA approximation via EMA
    teeth_1w = close_1w.ewm(alpha=1/8, adjust=False).mean().values
    lips_1w = close_1w.ewm(alpha=1/5, adjust=False).mean().values
    
    # Align to 1d timeframe (wait for completed 1w bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Alligator warmup and volume MA warmup
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime detection
        bull_alligator = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        bear_alligator = (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i])
        
        if position == 0:
            # Long: bull regime + price above lips + volume spike
            long_signal = bull_alligator and (close[i] > lips_aligned[i]) and volume_spike[i]
            
            # Short: bear regime + price below jaw + volume spike
            short_signal = bear_alligator and (close[i] < jaw_aligned[i]) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: regime change to not bull (chop or bear) OR price below lips
            if not bull_alligator or close[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: regime change to not bear (chop or bull) OR price above jaw
            if not bear_alligator or close[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_HTF_Alligator_Trend_Regime_VolumeFilter"
timeframe = "1d"
leverage = 1.0