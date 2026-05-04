#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian upper channel with 1d uptrend and volume spike.
# Short when price breaks below Donchian lower channel with 1d downtrend and volume spike.
# Donchian channels provide clear structural breaks, EMA50 filters counter-trend trades,
# volume confirmation ensures breakout strength. Designed for 12-37 trades/year on 12h.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "12h_Donchian20_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 12h data (20-period)
    lookback = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        window_high = high[i - lookback + 1:i + 1]
        window_low = low[i - lookback + 1:i + 1]
        dc_upper[i] = np.max(window_high)
        dc_lower[i] = np.min(window_low)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian upper AND 1d uptrend AND volume spike
            if (close[i] > dc_upper[i] and 
                close[i] > ema_50_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian lower AND 1d downtrend AND volume spike
            elif (close[i] < dc_lower[i] and 
                  close[i] < ema_50_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR 1d trend turns down
            if (close[i] < dc_lower[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR 1d trend turns up
            if (close[i] > dc_upper[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals