#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
# Uses 4h primary timeframe for balanced trade frequency (~25-40 trades/year) to minimize fee drag.
# Donchian channel provides clear breakout levels, filtered by 1d EMA200 trend and volume spikes (>1.5x 20-bar MA).
# Designed to work in both bull and bear markets by following the 1d trend while using Donchian breakouts as entry signals.
# Target: 75-200 total trades over 4 years (19-50/year). Size: 0.30.

name = "4h_Donchian20_Breakout_1dEMA200_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA200 (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    # Calculate 4h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # EMA200 needs 200 bars, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA200 direction
        price_above_ema = close[i] > ema_200_1d_aligned[i]
        price_below_ema = close[i] < ema_200_1d_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: reverse signal or close below/above midpoint
        midpoint = (donchian_high[i] + donchian_low[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals