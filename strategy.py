#!/usr/bin/env python3

"""
Hypothesis: 6-hour Donchian(20) breakout with daily trend filter and volume confirmation.
This strategy captures medium-term breakouts in trending markets while avoiding false breakouts
in ranging conditions. The daily trend filter (price above/below EMA50) ensures we trade with
the higher timeframe momentum, and volume spikes confirm institutional participation.
Designed to work in both bull and bear markets by adapting to the daily trend.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-period high with bullish daily trend and volume spike
            if (close[i] > high_max_20[i] and 
                close[i] > ema50_aligned[i] and  # Bullish trend: price above EMA50
                volume[i] > 1.8 * vol_avg_20[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low with bearish daily trend and volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < ema50_aligned[i] and  # Bearish trend: price below EMA50
                  volume[i] > 1.8 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Donchian level or contrary trend
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below 20-period low OR turns bearish
                if (close[i] < low_min_20[i] or 
                    close[i] < ema50_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above 20-period high OR turns bullish
                if (close[i] > high_max_20[i] or 
                    close[i] > ema50_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_20_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0