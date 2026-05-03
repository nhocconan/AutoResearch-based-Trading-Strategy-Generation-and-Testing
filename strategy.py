#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Donchian breakouts capture momentum in trending markets. The 12h EMA50 ensures we only
# trade in the direction of the higher timeframe trend, reducing whipsaws. Volume spike
# confirms conviction. Designed for 20-40 trades/year on 4h to minimize fee drag.
# Works in both bull and bear markets by aligning with HTF trend and requiring volume.

name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient warmup for Donchian
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period) using data up to current bar
        lookback = min(20, i+1)
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high
        breakout_down = close[i] < lowest_low
        
        if position == 0:
            # Long: Donchian breakout up in 12h uptrend with volume spike
            if breakout_up and ema_50_12h_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down in 12h downtrend with volume spike
            elif breakout_down and ema_50_12h_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakout down or loses 12h uptrend
            if breakout_down or ema_50_12h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout up or loses 12h downtrend
            if breakout_up or ema_50_12h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals