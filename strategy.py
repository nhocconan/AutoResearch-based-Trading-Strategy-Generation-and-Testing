#!/usr/bin/env python3
"""
6h Williams Alligator + Chop Filter + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction on 6h timeframe.
Choppiness Index filter ensures we only trade in trending regimes (CHOP < 38.2).
Volume spike confirms momentum. Works in bull/bear by only taking Alligator-aligned trades.
Target: 50-150 total trades over 4 years (12-37/year). Discrete sizing 0.0, ±0.25 to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Alligator and Chop calculation (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(data, period):
        """Smoothed Moving Average"""
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(df_6h['close'].values, 13)
    teeth = smma(df_6h['close'].values, 8)
    lips = smma(df_6h['close'].values, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align Alligator lines to 6h timeframe (already on 6h, but using align for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Calculate Choppiness Index on 6h
    def choppiness_index(high, low, close, period=14):
        """Choppiness Index: 0 = trending, 100 = ranging"""
        atr = np.zeros_like(close)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        
        # Wilder's smoothing for ATR
        atr_smoothed = np.full_like(atr, np.nan)
        if len(atr) < period:
            return atr_smoothed * 100  # Return zeros scaled
            
        atr_smoothed[period-1] = np.mean(atr[1:period])
        for i in range(period, len(atr)):
            atr_smoothed[i] = (atr_smoothed[i-1] * (period-1) + atr[i]) / period
        
        # Calculate highest high and lowest low over period
        hh = np.zeros_like(high)
        ll = np.zeros_like(low)
        for i in range(len(high)):
            if i < period-1:
                hh[i] = np.nan
                ll[i] = np.nan
            else:
                hh[i] = np.max(high[i-period+1:i+1])
                ll[i] = np.min(low[i-period+1:i+1])
        
        # Chop = 100 * log10(sum(ATR) / (HH - LL)) / log10(period)
        sum_atr = np.zeros_like(close)
        for i in range(len(close)):
            if i < period-1:
                sum_atr[i] = np.nan
            else:
                sum_atr[i] = np.sum(atr_smoothed[i-period+1:i+1])
        
        chop = np.full_like(close, np.nan)
        mask = (~np.isnan(sum_atr)) & (~np.isnan(hh)) & (~np.isnan(ll)) & ((hh - ll) > 0)
        chop[mask] = 100 * np.log10(sum_atr[mask] / (hh[mask] - ll[mask])) / np.log10(period)
        return chop
    
    chop = choppiness_index(df_6h['high'].values, df_6h['low'].values, df_6h['close'].values, 14)
    chop_aligned = align_htf_to_ltf(prices, df_6h, chop)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator and Chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        uptrend = (lips_val > teeth_val) and (teeth_val > jaw_val)
        downtrend = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Chop filter: only trade when trending (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Look for entry signals
            # Long: Alligator uptrend AND trending regime AND volume spike
            long_entry = uptrend and trending_regime and vol_spike
            # Short: Alligator downtrend AND trending regime AND volume spike
            short_entry = downtrend and trending_regime and vol_spike
            
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
            # Exit: Alligator loses uptrend alignment OR chop rises above 50 (ranging)
            if not uptrend or chop_val >= 50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator loses downtrend alignment OR chop rises above 50 (ranging)
            if not downtrend or chop_val >= 50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ChopFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0