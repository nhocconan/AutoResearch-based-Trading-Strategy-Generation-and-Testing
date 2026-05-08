#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Liquidity_Grab_Reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for liquidity zones
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day high/low for liquidity grab identification
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    
    # Align daily levels to 6h timeframe
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    
    # Volume confirmation: 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(prev_day_high_aligned[i]) or np.isnan(prev_day_low_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Liquidity grab detection
        # Bullish setup: price breaks below prev day low then reverses above it
        # Bearish setup: price breaks above prev day high then reverses below it
        
        if position == 0:
            # Long: bullish liquidity grab reversal
            # Price was below prev day low 2 periods ago, now above it
            if (i >= 2 and 
                low[i-2] < prev_day_low_aligned[i-2] and  # Was below
                close[i] > prev_day_low_aligned[i] and      # Now above
                vol_ratio[i] > 1.8):                        # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: bearish liquidity grab reversal
            # Price was above prev day high 2 periods ago, now below it
            elif (i >= 2 and 
                  high[i-2] > prev_day_high_aligned[i-2] and  # Was above
                  close[i] < prev_day_high_aligned[i] and      # Now below
                  vol_ratio[i] > 1.8):                        # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below entry area or volume dries up
            if (close[i] < prev_day_low_aligned[i] * 0.995 or  # Slight buffer
                vol_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above entry area or volume dries up
            if (close[i] > prev_day_high_aligned[i] * 1.005 or  # Slight buffer
                vol_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals