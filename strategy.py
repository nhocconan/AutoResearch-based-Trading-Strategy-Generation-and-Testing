#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike
# Donchian breakout captures strong directional moves. Combined with 1d EMA34 trend filter 
# and volume confirmation, this strategy enters only when breakout aligns with daily trend 
# and has institutional volume support. Works in both bull and bear markets by following 
# the higher-timeframe trend. Target: 50-150 trades over 4 years (12-37/year) on 12h.

name = "12h_Donchian20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    if len(high) < 20 or len(low) < 20 or len(close) < 20:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 20-period average (~10 days for 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and 1d EMA)
    start_idx = max(34, 20)  # 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band with volume spike AND price > 1d EMA34 (bullish trend)
            if (close[i] > highest_high[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower band with volume spike AND price < 1d EMA34 (bearish trend)
            elif (close[i] < lowest_low[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian middle OR price below 1d EMA34 (trend change)
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < donchian_middle or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle OR price above 1d EMA34 (trend change)
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > donchian_middle or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals