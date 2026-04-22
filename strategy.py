#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with 1-day trend filter and volume spike.
Long when price > Alligator teeth (green line) with rising 1-day EMA34 and volume spike.
Short when price < Alligator teeth with falling 1-day EMA34 and volume spike.
Exit when price crosses back below/above teeth.
Williams Alligator identifies trend presence and direction; 1-day EMA34 filters for higher timeframe trend;
volume spike confirms institutional participation. Designed for low trade frequency by requiring
multiple confirmations. Works in both bull and bear markets by following the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Williams Alligator (13,8,5 SMAs shifted)
    # Jaw (blue): 13-period SMA, shifted 8 bars
    # Teeth (red): 8-period SMA, shifted 5 bars
    # Lips (green): 5-period SMA, shifted 3 bars
    # We use the teeth (8-period SMA shifted 5) as the main trend line
    
    # Calculate SMAs
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    
    # Apply shifts: Jaw (13) shifted 8, Teeth (8) shifted 5, Lips (5) shifted 3
    jaw = np.roll(sma_13, 8)
    teeth = np.roll(sma_8, 5)
    lips = np.roll(sma_5, 3)
    
    # Fill shifted values with NaN for invalid periods
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after enough data for SMA13
        # Skip if data not ready
        if (np.isnan(teeth[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price > teeth with rising 1-day EMA34 and volume spike
            if (close[i] > teeth[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price < teeth with falling 1-day EMA34 and volume spike
            elif (close[i] < teeth[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back below/above teeth
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below teeth
                if close[i] < teeth[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above teeth
                if close[i] > teeth[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Williams_Alligator_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0