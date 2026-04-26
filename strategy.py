#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter
Hypothesis: On 1h timeframe, enter long when price breaks above Camarilla R1 level AND 4h trend is up (close > EMA50) AND volume > 1.5x 20-period average volume AND within 08-20 UTC session. Enter short when price breaks below Camarilla S1 level AND 4h trend is down (close < EMA50) AND volume > 1.5x 20-period average volume AND within 08-20 UTC session. Exit on trend reversal or price retracing to Camarilla midpoint. Uses discrete sizing (0.0, ±0.20) to limit fee drag. Target: 15-37 trades/year on 1h timeframe.
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels (using daily for structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous 1d bar
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
    r1 = prev_close_1d + 1.1 * camarilla_range / 12
    s1 = prev_close_1d - 1.1 * camarilla_range / 12
    r3 = prev_close_1d + 1.1 * camarilla_range / 4
    s3 = prev_close_1d - 1.1 * camarilla_range / 4
    mid = (r1 + s1) / 2  # Camarilla midpoint for exit
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    mid_aligned = align_htf_to_ltf(prices, df_1d, mid)
    
    # Volume confirmation: fixed threshold of 1.5x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup and volume MA warmup
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        # 4h trend filter
        trend_uptrend = close[i] > ema_50_4h_aligned[i]
        trend_downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume spike + 4h uptrend + session
            long_signal = breakout_up and volume_spike[i] and trend_uptrend
            
            # Short: breakout below S1 + volume spike + 4h downtrend + session
            short_signal = breakout_down and volume_spike[i] and trend_downtrend
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: trend change to downtrend OR price retracing to Camarilla midpoint
            if not trend_uptrend or close[i] < mid_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: trend change to uptrend OR price retracing to Camarilla midpoint
            if not trend_downtrend or close[i] > mid_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0