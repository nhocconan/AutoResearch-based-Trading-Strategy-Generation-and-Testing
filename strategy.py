#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike confirmation.
# Long when price breaks above Donchian upper channel AND 12h EMA50 rising AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower channel AND 12h EMA50 falling AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25. Target: 20-50 trades/year on 4h.
# Donchian channels provide clear structure, 12h EMA filters counter-trend trades, volume spike confirms conviction.
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend) by aligning with higher timeframe.

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    # Upper channel = highest high over past 20 bars
    # Lower channel = lowest low over past 20 bars
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_dc_upper = donchian_upper[i]
        curr_dc_lower = donchian_lower[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_volume_ma = volume_ma_20[i]
        curr_volume_spike = volume_spike[i]
        
        # Donchian breakout conditions
        breakout_up = curr_close > curr_dc_upper
        breakout_down = curr_close < curr_dc_lower
        
        # 12h EMA trend: rising if current > previous, falling if current < previous
        if i > start_idx:
            prev_ema_12h = ema_50_12h_aligned[i-1]
            ema_rising = curr_ema_12h > prev_ema_12h
            ema_falling = curr_ema_12h < prev_ema_12h
        else:
            ema_rising = False
            ema_falling = False
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND 12h EMA rising AND volume spike
            if (breakout_up and 
                ema_rising and 
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND 12h EMA falling AND volume spike
            elif (breakout_down and 
                  ema_falling and 
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Donchian breakdown OR 12h EMA turns falling
            if (breakout_down or not ema_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Donchian breakout up OR 12h EMA turns rising
            if (breakout_up or not ema_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals