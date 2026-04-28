#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1-day ATR filter and volume confirmation.
# Donchian(20) breakout provides clear entry/exit signals based on price channels.
# ATR(14) filter ensures we only trade during sufficient volatility periods (ATR > 0.5 * ATR(50)).
# Volume confirmation requires volume > 1.5x 20-period average to ensure participation.
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear markets by filtering for volatile breakouts.

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
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR calculation using Wilder's smoothing
    atr_14 = np.zeros_like(tr)
    atr_14[0] = tr[0]
    for i in range(1, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR(50) for volatility regime filter
    atr_50 = np.zeros_like(tr)
    atr_50[:49] = np.nan
    for i in range(49, len(tr)):
        if i == 49:
            atr_50[i] = np.mean(tr[0:50])
        else:
            atr_50[i] = (atr_50[i-1] * 49 + tr[i]) / 50
    
    # Align ATR indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Donchian channels (20-period) on 4h data
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    
    for i in range(len(high)):
        if i < 19:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) > 0.5 * ATR(50) ensures sufficient volatility
        volatility_filter = atr_14_aligned[i] > 0.5 * atr_50_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Entry conditions with volume and volatility confirmation
        long_entry = long_breakout and volatility_filter and volume_filter[i]
        short_entry = short_breakout and volatility_filter and volume_filter[i]
        
        # Exit conditions: opposite Donchian breakout or volatility collapse
        long_exit = (not volatility_filter) or (close[i] < lowest_low[i])
        short_exit = (not volatility_filter) or (close[i] > highest_high[i])
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_1dATR_VolFilter_Volume"
timeframe = "4h"
leverage = 1.0