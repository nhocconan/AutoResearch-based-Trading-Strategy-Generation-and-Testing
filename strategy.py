#!/usr/bin/env python3
"""
4h_ema_cross_volume_v3
Hypothesis: On 4h timeframe, enter long when fast EMA crosses above slow EMA with above-average volume and price above 100-period SMA (bullish trend), enter short when fast EMA crosses below slow EMA with above-average volume and price below 100-period SMA (bearish trend). Exit when EMA cross reverses. Uses 1d EMA trend filter to avoid counter-trend trades. Designed for 20-30 trades/year to minimize fee drag while capturing major trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema_cross_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMAs
    ema_fast = pd.Series(close).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Calculate 100-period SMA for trend filter
    sma_100 = pd.Series(close).rolling(window=100, min_periods=100).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA for trend filter (avoid counter-trend trades)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_fast_1d = pd.Series(close_1d).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema_slow_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # 1d EMA crossover signal (1 = bullish, -1 = bearish)
    ema_cross_1d = np.where(ema_fast_1d > ema_slow_1d, 1, -1)
    
    # Align indicators to 4h timeframe
    ema_cross_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_cross_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if data not available
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(sma_100[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_cross_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: EMA cross turns bearish
            if ema_fast[i] < ema_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA cross turns bullish
            if ema_fast[i] > ema_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: EMA bullish cross with price above SMA100 and 1d trend bullish
                if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1] and close[i] > sma_100[i] and ema_cross_1d_aligned[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Short: EMA bearish cross with price below SMA100 and 1d trend bearish
                elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1] and close[i] < sma_100[i] and ema_cross_1d_aligned[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals