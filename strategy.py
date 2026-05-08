#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout (40-period) with 1w EMA20 trend filter and volume confirmation.
# Long when price breaks above 40-period Donchian high and 1w EMA20 rising, with volume spike.
# Short when price breaks below 40-period Donchian low and 1w EMA20 falling, with volume spike.
# Exit when price crosses opposite Donchian band or EMA20 direction changes.
# Uses 6h for Donchian channels and volume, 1w for EMA20 trend.
# Designed to capture trends with controlled frequency to avoid fee drag.
# Target: 10-30 trades/year to stay within profitable range.

name = "6h_Donchian_Breakout_1wEMA20_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels and volume (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 40:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (40-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    donchian_high = pd.Series(high_6h).rolling(window=40, min_periods=40).max().values
    donchian_low = pd.Series(low_6h).rolling(window=40, min_periods=40).min().values
    
    # Calculate 6h 20-period average volume for volume filter
    vol_ma_20 = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    ema_20 = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Calculate EMA20 slope for trend direction (rising/falling)
    ema_20_prev = np.roll(ema_20, 1)
    ema_20_prev[0] = ema_20[0]
    ema_20_rising = ema_20 > ema_20_prev
    ema_20_falling = ema_20 < ema_20_prev
    
    # Align all indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    ema_20_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_20_rising)
    ema_20_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_20_falling)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_20_aligned[i]) or \
           np.isnan(ema_20_rising_aligned[i]) or np.isnan(ema_20_falling_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 6h volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 6h bar's volume
            idx_6h = 0
            while idx_6h < len(df_6h) and df_6h.iloc[idx_6h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_6h += 1
            idx_6h -= 1  # last completed 6h bar
            
            if idx_6h >= 0:
                vol_6h_current = df_6h.iloc[idx_6h]['volume']
                vol_filter = vol_6h_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout + trend + volume
            # Long when price breaks above Donchian high, EMA20 rising, with volume spike
            long_condition = (close[i] > donchian_high_aligned[i]) and ema_20_rising_aligned[i] and vol_filter
            # Short when price breaks below Donchian low, EMA20 falling, with volume spike
            short_condition = (close[i] < donchian_low_aligned[i]) and ema_20_falling_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low or EMA20 starts falling
            if (close[i] < donchian_low_aligned[i]) or (not ema_20_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high or EMA20 starts rising
            if (close[i] > donchian_high_aligned[i]) or (not ema_20_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals