#!/usr/bin/env python3

"""
Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
Donchian breakouts capture momentum in trending markets while the daily trend filter
prevents counter-trend trades. Volume spikes confirm institutional interest.
This should work in both bull and bear regimes by adapting to the daily trend.
Target: 20-50 trades/year per symbol.
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: breakout above upper band, price above daily EMA, volume spike
            if (high[i] > high_20[i-1] and  # Breakout above 20-period high
                close[i] > ema_34_aligned[i] and  # Price above daily EMA
                volume[i] > 1.5 * vol_avg_20[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower band, price below daily EMA, volume spike
            elif (low[i] < low_20[i-1] and   # Breakdown below 20-period low
                  close[i] < ema_34_aligned[i] and  # Price below daily EMA
                  volume[i] > 1.5 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite breakout or return to midline
            exit_signal = False
            
            if position == 1:
                # Exit long: breakdown below lower band OR price returns to midline
                if (low[i] < low_20[i-1] or 
                    close[i] < (high_20[i-1] + low_20[i-1]) / 2):
                    exit_signal = True
            else:  # position == -1
                # Exit short: breakout above upper band OR price returns to midline
                if (high[i] > high_20[i-1] or 
                    close[i] > (high_20[i-1] + low_20[i-1]) / 2):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0