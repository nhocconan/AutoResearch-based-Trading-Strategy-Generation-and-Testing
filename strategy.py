#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d volume spike and weekly trend filter.
Long when Alligator jaws < teeth < lips (bullish alignment) AND volume > 2.0x 20-period average AND weekly close > weekly EMA50.
Short when Alligator jaws > teeth > lips (bearish alignment) AND volume > 2.0x 20-period average AND weekly close < weekly EMA50.
Exit when Alligator lines cross (jaws crosses teeth) or weekly trend changes.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Williams Alligator (SMAs of median price) provides trend direction with built-in smoothing.
Weekly EMA50 filter ensures alignment with higher timeframe trend, reducing counter-trend trades.
Volume confirmation ensures institutional participation in breakouts.
Designed to work in both bull and bear markets by requiring weekly trend alignment.
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
    
    # Load 1d data for Williams Alligator - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate median price for 1d
    median_price_1d = (high_1d + low_1d) / 2.0
    
    # Williams Alligator: three SMAs of median price
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars  
    # Lips: 5-period SMA, shifted 3 bars
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for Alligator and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment (jaws < teeth < lips) AND volume spike AND weekly uptrend
            if (jaw_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment (jaws > teeth > lips) AND volume spike AND weekly downtrend
            elif (jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator lines cross (jaws crosses teeth) - trend change signal
            if position == 1 and jaw_1d_aligned[i] >= teeth_1d_aligned[i]:
                exit_signal = True
            elif position == -1 and jaw_1d_aligned[i] <= teeth_1d_aligned[i]:
                exit_signal = True
            
            # Secondary exit: Weekly trend changes
            elif position == 1 and close[i] < ema50_1w_aligned[i]:
                exit_signal = True
            elif position == -1 and close[i] > ema50_1w_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dVolumeSpike_1wEMA50"
timeframe = "12h"
leverage = 1.0