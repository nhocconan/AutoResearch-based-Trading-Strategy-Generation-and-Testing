#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R1 level from previous 1d candle AND 1d trend is up (close > EMA34) AND volume > 2.0x 20-period average volume. Enter short when price breaks below Camarilla S1 level AND 1d trend is down (close < EMA34) AND volume > 2.0x 20-period average volume. Exit on trend reversal or price retracing to Camarilla midpoint. Uses discrete sizing (0.0, ±0.25) to limit fee drag. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla R1 and S1 levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_raw = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d_raw, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12  # Camarilla R1
    s1 = prev_close_1d - 1.1 * camarilla_range / 12  # Camarilla S1
    mid = (r1 + s1) / 2  # Camarilla midpoint for exit
    
    # Align Camarilla levels and EMA34 to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    mid_aligned = align_htf_to_ltf(prices, df_1d, mid)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: fixed threshold of 2.0x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and volume MA warmup
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume spike + 1d uptrend
            long_signal = breakout_up and volume_spike[i] and trend_uptrend
            
            # Short: breakout below S1 + volume spike + 1d downtrend
            short_signal = breakout_down and volume_spike[i] and trend_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend change to downtrend OR price retracing to Camarilla midpoint
            if not trend_uptrend or close[i] < mid_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend change to uptrend OR price retracing to Camarilla midpoint
            if not trend_downtrend or close[i] > mid_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0