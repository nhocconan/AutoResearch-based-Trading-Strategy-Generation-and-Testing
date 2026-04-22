#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d trend filter and volume confirmation.
Long when price breaks above Donchian high with bullish 1d trend (price > EMA50) and volume spike.
Short when price breaks below Donchian low with bearish 1d trend (price < EMA50) and volume spike.
Exit when price returns to Donchian mid-point.
Designed for low trade frequency (15-30/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after lookback periods
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
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
            # Long: Price breaks above Donchian high with bullish 1d trend and volume spike
            if (close[i] > donch_high[i] and 
                close[i] > ema50_aligned[i] and  # Bullish trend: price above EMA50
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with bearish 1d trend and volume spike
            elif (close[i] < donch_low[i] and 
                  close[i] < ema50_aligned[i] and  # Bearish trend: price below EMA50
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Donchian mid-point
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to mid-point
                if close[i] <= donch_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to mid-point
                if close[i] >= donch_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_20_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0
#%%