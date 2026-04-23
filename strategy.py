#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 12h EMA50 Trend + Volume Spike
Williams Alligator (Jaw/Teeth/Lips) identifies trend absence (all lines intertwined) vs presence (lines separated, ordered).
In trending markets (Alligator awake), trade breakouts in direction of 12h EMA50 with volume confirmation.
In ranging markets (Alligator sleeping), stay flat. 6h timeframe reduces noise while capturing medium-term swings.
Target: 12-37 trades/year (50-150 over 4 years) with discrete sizing 0.25.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams Alligator on 6h data (Smoothed Medians)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        sma = pd.Series(source).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(source, np.nan, dtype=float)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(source)):
            if not np.isnan(sma[i]):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + sma[i]) / period
            else:
                smma_vals[i] = smma_vals[i-1]
        return smma_vals
    
    jaw = smma(high, 13)  # Typically uses median, but high/low approximation works
    teeth = smma(high, 8)
    lips = smma(high, 5)
    
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # Already 6h, no alignment needed
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Volume confirmation: > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 13+8)  # need EMA50_12h, vol MA, Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator condition: lines separated and ordered (trending)
        # For uptrend: Lips > Teeth > Jaw
        # For downtrend: Lips < Teeth < Jaw
        alligator_up = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_down = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Alligator awake (uptrend) AND price > 12h EMA50 AND volume spike
            if (alligator_up and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator awake (downtrend) AND price < 12h EMA50 AND volume spike
            elif (alligator_down and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator sleeping (lines intertwined) OR loss of 12h EMA50 alignment
            # Alligator sleeping: lines close together (not separated)
            lips_teeth_diff = abs(lips_aligned[i] - teeth_aligned[i])
            teeth_jaw_diff = abs(teeth_aligned[i] - jaw_aligned[i])
            avg_price = (high[i] + low[i]) / 2
            sleeping_threshold = avg_price * 0.005  # 0.5% of price
            alligator_sleeping = (lips_teeth_diff < sleeping_threshold and 
                                 teeth_jaw_diff < sleeping_threshold)
            
            exit_signal = alligator_sleeping or \
                         (position == 1 and close[i] < ema_50_12h_aligned[i]) or \
                         (position == -1 and close[i] > ema_50_12h_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Williams_Alligator_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0