#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-bar Donchian high AND 1w EMA50 is rising AND volume spike
# Short when price breaks below 20-bar Donchian low AND 1w EMA50 is falling AND volume spike
# Donchian channels provide clear breakout levels; weekly EMA50 filters for major trend direction
# Volume spike confirms institutional participation in the breakout
# Works in bull (breakouts with buying volume) and bear (breakdowns with selling volume)
# Timeframe: 6h (primary timeframe as required)
# Target: 50-150 total trades over 4 years (12-37/year) to balance signal quality and fee drag

name = "6h_Donchian20_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    if len(close_1w) >= 50:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        # Rising/falling: current > previous bar
        ema_50_rising = np.zeros(len(close_1w), dtype=bool)
        ema_50_falling = np.zeros(len(close_1w), dtype=bool)
        ema_50_rising[1:] = ema_50_1w[1:] > ema_50_1w[:-1]
        ema_50_falling[1:] = ema_50_1w[1:] < ema_50_1w[:-1]
        # First bar: no trend
        ema_50_rising[0] = False
        ema_50_falling[0] = False
    else:
        ema_50_rising = np.zeros(len(close_1w), dtype=bool)
        ema_50_falling = np.zeros(len(close_1w), dtype=bool)
    
    # Align 1w EMA50 trend to 6h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling)
    
    # Calculate 6h Donchian channels (20-period)
    if len(high) >= 20 and len(low) >= 20:
        # Donchian high: highest high over last 20 periods
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian low: lowest low over last 20 periods
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Volume confirmation on 6h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND 1w EMA50 rising AND volume spike
            if (close[i] > donchian_high[i] and 
                ema_50_rising_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND 1w EMA50 falling AND volume spike
            elif (close[i] < donchian_low[i] and 
                  ema_50_falling_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR 1w EMA50 falling
            if close[i] < donchian_low[i] or ema_50_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR 1w EMA50 rising
            if close[i] > donchian_high[i] or ema_50_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals