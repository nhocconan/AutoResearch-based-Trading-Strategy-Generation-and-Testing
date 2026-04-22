#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
    # Donchian breakouts capture momentum in trending markets.
    # 1d EMA50 filter ensures trades align with daily trend to avoid counter-trend whipsaws.
    # Volume spike confirms institutional participation.
    # Targets ~20-30 trades/year to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian calculation (same timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    # Using rolling window on 4h data
    high_roll = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_roll)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_roll)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-period high + above 1d EMA50 (uptrend) + volume spike
            if close[i] > high_20_aligned[i] and close[i] > ema50_1d_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low + below 1d EMA50 (downtrend) + volume spike
            elif close[i] < low_20_aligned[i] and close[i] < ema50_1d_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle of channel or trend reverses
            if position == 1:
                if close[i] < (high_20_aligned[i] + low_20_aligned[i]) / 2 or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > (high_20_aligned[i] + low_20_aligned[i]) / 2 or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_Volume_Session_v1"
timeframe = "4h"
leverage = 1.0