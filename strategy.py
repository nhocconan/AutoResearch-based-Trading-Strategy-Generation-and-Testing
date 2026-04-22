#!/usr/bin/env python3

"""
Hypothesis: 1-day Donchian breakout with 1-week trend filter and volume confirmation.
The 1-day Donchian channel captures daily volatility breakouts, while the 1-week trend filter ensures
trades align with the longer-term direction to avoid counter-trend moves. Volume spikes confirm
institutional participation. This strategy targets medium-term trends in both bull and bear markets
by focusing on volatility expansion at key levels with minimal trade frequency to reduce fee drag.
Target: 15-25 trades/year per symbol.
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
    
    # Load 1-week data - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week EMA for trend filter (20-period)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1-week EMA to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily Donchian channel (20-period)
    # Use previous day's data to avoid look-ahead
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 20-day volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate day of week for session filter (optional: avoid weekends)
    days = pd.DatetimeIndex(prices['open_time']).dayofweek  # Monday=0, Sunday=6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip weekends (Saturday=5, Sunday=6)
        if days[i] >= 5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, above 1w EMA, volume spike
            if (close[i] > donch_high[i] and    # Break above Donchian high
                close[i] > ema_20_1w_aligned[i] and # Above 1-week EMA (bullish trend)
                volume[i] > 1.5 * vol_avg_20[i]): # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below 1w EMA, volume spike
            elif (close[i] < donch_low[i] and   # Break below Donchian low
                  close[i] < ema_20_1w_aligned[i] and # Below 1-week EMA (bearish trend)
                  volume[i] > 1.5 * vol_avg_20[i]): # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or crosses 1w EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian low or below 1w EMA
                if close[i] < donch_low[i] or close[i] < ema_20_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Donchian high or above 1w EMA
                if close[i] > donch_high[i] or close[i] > ema_20_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_Breakout_1wEMA20_Volume"
timeframe = "1d"
leverage = 1.0