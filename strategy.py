# 4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
# Hypothesis: Use weekly trend filter with daily Camarilla R1/S1 breakouts on 4h.
# The weekly trend ensures we only trade in the direction of the higher timeframe,
# reducing false breakouts in choppy markets. Volume spike confirms breakout strength.
# This combination has shown strong performance in both bull and bear markets.

#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    
    # Camarilla R1 and S1 levels (most significant for intraday trading)
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Weekly trend: 50-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50).mean().values
    weekly_trend_up = ema_50_1w > np.roll(ema_50_1w, 1)  # rising EMA
    weekly_trend_up = np.roll(weekly_trend_up, 1)  # shift forward to avoid look-ahead
    weekly_trend_up[0] = False  # first value has no previous
    
    # Align weekly trend to 4h timeframe
    weekly_trend_up_4h = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(weekly_trend_up_4h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND weekly trend up AND volume spike
            if close[i] > r1_4h[i] and weekly_trend_up_4h[i] > 0.5 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND weekly trend down AND volume spike
            elif close[i] < s1_4h[i] and weekly_trend_up_4h[i] < 0.5 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 OR weekly trend turns down
            if close[i] < s1_4h[i] or weekly_trend_up_4h[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R1 OR weekly trend turns up
            if close[i] > r1_4h[i] or weekly_trend_up_4h[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals