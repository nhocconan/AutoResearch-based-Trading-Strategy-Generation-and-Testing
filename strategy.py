#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw=TEETH=LIPS SMMA) identifies trend absence (sleeping) vs presence (awake).
When price > 1d EMA50 (uptride) and Alligator jaws (TEETH) crossover up with LIPS above TEETH = bullish momentum.
When price < 1d EMA50 (downtrend) and jaws crossover down with LIPS below TEETH = bearish momentum.
Volume spike confirms participation. Works in bull (buy dips) and bear (sell rallies) by trading with 1d trend.
6h timeframe targets 12-37 trades/year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (Williams Alligator uses SMMA)"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    smma_vals = np.empty_like(series, dtype=float)
    smma_vals[:] = np.nan
    smma_vals[period-1] = np.mean(series[:period])
    for i in range(period, len(series)):
        smma_vals[i] = (smma_vals[i-1] * (period-1) + series[i]) / period
    return smma_vals

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
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
    start_idx = max(20, 13)  # volume MA, Alligator jaws
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_aligned[i]
        downtrend = curr_close < ema_50_aligned[i]
        
        # Alligator signals: jaws crossover with confirmation
        # Bullish: teeth crosses above jaw AND lips above teeth
        bullish_cross = (curr_teeth > curr_jaw) and (curr_lips > curr_teeth)
        # Bearish: teeth crosses below jaw AND lips below teeth
        bearish_cross = (curr_teeth < curr_jaw) and (curr_lips < curr_teeth)
        
        if position == 0:
            # Look for entry signals
            # Long: Bullish crossover AND uptrend AND volume spike
            long_entry = bullish_cross and uptrend and vol_spike
            # Short: Bearish crossover AND downtrend AND volume spike
            short_entry = bearish_cross and downtrend and vol_spike
            
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
            # Exit: Bearish crossover (teeth below jaw) OR loss of uptrend
            if (curr_teeth < curr_jaw) or (curr_close < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Bullish crossover (teeth above jaw) OR loss of downtrend
            if (curr_teeth > curr_jaw) or (curr_close > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_VolumeSpike_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0