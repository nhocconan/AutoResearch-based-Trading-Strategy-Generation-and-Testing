#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12-hour Donchian channel breakout with 1-day trend filter and volume confirmation
    # Donchian(20) on 12h: breakout above upper channel = long, below lower = short
    # 1-day EMA34 filters trend: only long in uptrend, short in downtrend
    # Volume spike confirms breakout strength
    # Targets 15-25 trades/year to minimize fee drag on 12h timeframe
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian calculations
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 12h
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Align indicators to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_spike_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper Donchian + uptrend (price > 1d EMA34) + volume spike
            if close[i] > upper_20_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower Donchian + downtrend (price < 1d EMA34) + volume spike
            elif close[i] < lower_20_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price reverts to middle of Donchian channel or trend reverses
            mid_channel = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
            if position == 1:
                if close[i] < mid_channel or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > mid_channel or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_20_Breakout_1dEMA34_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0