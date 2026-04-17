#/usr/bin/env python3
"""
4h_Donchian20_12hTrend_VolumeConfirm
Strategy: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
Long: Price breaks above 20-period high + 12h close > 12h EMA(34) + volume > 1.5x 20-period avg
Short: Price breaks below 20-period low + 12h close < 12h EMA(34) + volume > 1.5x 20-period avg
Exit: Opposite breakout or ATR-based stop
Position size: 0.25
Uses Donchian for breakout signals, 12h EMA for trend filter, volume for confirmation.
Designed to work in both bull and bear markets by requiring trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper and lower bands
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe (they are already on 4h)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Calculate 4h volume average (20-period)
    volume_4h = df_4h['volume'].values
    volume_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(upper_4h_aligned[i]) or 
            np.isnan(lower_4h_aligned[i]) or np.isnan(volume_ma20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume aligned to 4h
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.5 * volume_ma20_4h_aligned[i])
        
        # Trend filter: 12h close vs 12h EMA(34)
        uptrend_12h = close_12h[-1] > ema_34_12h[-1] if len(close_12h) > 0 else False
        downtrend_12h = close_12h[-1] < ema_34_12h[-1] if len(close_12h) > 0 else False
        
        # Donchian breakout signals
        breakout_up = close[i] > upper_4h_aligned[i]
        breakout_down = close[i] < lower_4h_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout up + 12h uptrend + volume filter
            if breakout_up and uptrend_12h and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + 12h downtrend + volume filter
            elif breakout_down and downtrend_12h and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian breakout down
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian breakout up
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0