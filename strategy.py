#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian breakout with 1-day trend filter and volume confirmation.
Breakouts above 20-period Donchian high (for longs) or below 20-period Donchian low (for shorts)
signal momentum. The 1-day EMA-34 ensures trades align with daily trend, reducing counter-trend trades.
Volume spikes confirm institutional participation. This targets high-probability momentum
in both bull and bear markets by focusing on volatility expansion at key levels.
Target: 15-30 trades/year per symbol.
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
    
    # Load 1d data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high (20-period rolling max)
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian low (20-period rolling min)
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 12h timeframe (already aligned, but use function for consistency)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Calculate 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_avg_aligned[i])):
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
            # Long: price breaks above 12h Donchian high, above 1d EMA, volume spike
            if (close[i] > donch_high_aligned[i] and    # Break above Donchian high
                close[i] > ema_34_aligned[i] and         # Above 1d EMA (bullish trend)
                volume[i] > 2.0 * vol_avg_aligned[i]):   # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low, below 1d EMA, volume spike
            elif (close[i] < donch_low_aligned[i] and    # Break below Donchian low
                  close[i] < ema_34_aligned[i] and       # Below 1d EMA (bearish trend)
                  volume[i] > 2.0 * vol_avg_aligned[i]): # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or crosses 1d EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian low or below 1d EMA
                if close[i] < donch_low_aligned[i] or close[i] < ema_34_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Donchian high or above 1d EMA
                if close[i] > donch_high_aligned[i] or close[i] > ema_34_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0