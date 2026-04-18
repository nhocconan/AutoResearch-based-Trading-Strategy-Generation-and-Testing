#!/usr/bin/env python3
"""
12h Bull/Bear Regime Detection with Volume-Confirmed Breakouts
Strategy: Uses daily EMA crossovers to define bull/bear regimes, then enters
          breakout trades in the direction of the regime with volume confirmation.
          In bull regime: long on break above 12h high + volume, short on break below 12h low.
          In bear regime: short on break below 12h low + volume, long on break above 12h high.
          Exits when price reverses back to the regime's EMA or breaks opposite level.
          Designed to capture trends in both bull and bear markets while avoiding chop.
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
    
    # Get daily data for regime and breakout levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA21 and EMA50 for regime detection
    daily_close = df_1d['close'].values
    ema_21_1d = pd.Series(daily_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h high/low for breakout levels (use rolling window)
    high_12h = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_12h = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    # Volume spike detection (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Align daily EMAs to 12h timeframe
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(high_12h[i]) or 
            np.isnan(low_12h[i]) or
            np.isnan(ema_21_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_21 = ema_21_1d_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Determine regime: bull if EMA21 > EMA50, bear if EMA21 < EMA50
        is_bull = ema_21 > ema_50
        
        if position == 0:
            if is_bull:
                # Bull regime: long on break above 12h high with volume
                if price > high_12h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short on break below 12h low with volume (counter-trend but valid in strong moves)
                elif price < low_12h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Bear regime: short on break below 12h low with volume
                if price < low_12h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                # Long on break above 12h high with volume (counter-trend bounce)
                elif price > high_12h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks back below EMA21 (end of bull phase) or breaks 12h low
            if price < ema_21 or price < low_12h[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks back above EMA21 (end of bear phase) or breaks 12h high
            if price > ema_21 or price > high_12h[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_BullBearRegime_Breakout_Volume"
timeframe = "12h"
leverage = 1.0