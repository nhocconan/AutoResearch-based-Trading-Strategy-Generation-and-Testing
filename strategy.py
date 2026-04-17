#!/usr/bin/env python3
"""
Hypothesis: 12h Elder Ray Index with 1d Weekly Trend Filter and Volume Spike.
Long when Elder Bull Power > 0, Bear Power < 0, volume > 1.5x average, and 1d close > 1d EMA50.
Short when Elder Bull Power < 0, Bear Power > 0, volume > 1.5x average, and 1d close < 1d EMA50.
Exit when Elder Bull Power and Bear Power converge (|Bull - Bear| < 0.1 * ATR) or volume drops.
Uses 12h for price/volume/Elder Ray, 1d for trend filter. Target: 50-150 total trades over 4 years.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Elder Ray Index (requires 13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate 12h ATR(14) for exit condition
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr14 = np.zeros(n)
    atr14[14] = np.mean(tr[1:15])  # Seed with first 14 values
    for i in range(15, n):
        atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    # Calculate 12h volume MA(20) for volume spike filter
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Get 1d values (already aligned via index alignment in get_htf_data)
        ema50 = ema50_1d[i]
        vol_spike = volume_spike[i]
        
        # Elder Ray values
        bull = bull_power[i]
        bear = bear_power[i]
        atr = atr14[i]
        
        # Convergence condition: |Bull - Bear| < 0.1 * ATR
        convergence = abs(bull - bear) < (0.1 * atr)
        
        if position == 0:
            # Long: Bull > 0, Bear < 0, volume spike, uptrend (1d close > EMA50)
            if bull > 0 and bear < 0 and vol_spike and close[i] > ema50:
                signals[i] = 0.25
                position = 1
            # Short: Bull < 0, Bear > 0, volume spike, downtrend (1d close < EMA50)
            elif bull < 0 and bear > 0 and vol_spike and close[i] < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: convergence OR volume drops OR trend breaks
            if convergence or not vol_spike or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: convergence OR volume drops OR trend breaks
            if convergence or not vol_spike or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ElderRay_VolumeSpike_1dTrend"
timeframe = "12h"
leverage = 1.0