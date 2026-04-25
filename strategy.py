#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw=TEETH=LIPS SMMA) identifies trend absence (alligator sleeping) vs presence (alligator awakening).
In strong uptrends (price > 1d EMA34), Jaw > Teeth > Lips indicates bullish alignment for longs.
In strong downtrends (price < 1d EMA34), Jaw < Teeth < Lips indicates bearish alignment for shorts.
Volume spike confirms institutional participation. Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
6h timeframe targets 12-37 trades/year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (Williams Alligator uses SMMA)"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    result = np.full_like(series, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(series[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Williams Alligator components (using median price)
    median_price = (high + low) / 2.0
    jaw = smma(median_price, 13)   # Blue line
    teeth = smma(median_price, 8)  # Red line  
    lips = smma(median_price, 5)   # Green line
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 13)  # volume MA, Alligator jaws (13 period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        # Alligator alignment: Jaw > Teeth > Lips = bullish, Jaw < Teeth < Lips = bearish
        bullish_alignment = (curr_jaw > curr_teeth) and (curr_teeth > curr_lips)
        bearish_alignment = (curr_jaw < curr_teeth) and (curr_teeth < curr_lips)
        
        if position == 0:
            # Look for entry signals
            # Long: Bullish alignment AND uptrend AND volume spike
            long_entry = bullish_alignment and uptrend and vol_spike
            # Short: Bearish alignment AND downtrend AND volume spike
            short_entry = bearish_alignment and downtrend and vol_spike
            
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
            # Exit: Bullish alignment breaks OR loss of uptrend
            if not bullish_alignment or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Bearish alignment breaks OR loss of downtrend
            if not bearish_alignment or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_VolumeSpike_1dEMA34_Trend"
timeframe = "6h"
leverage = 1.0