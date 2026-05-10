#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: In 1h timeframe, Camarilla pivot levels (R1/S1) act as strong support/resistance.
# Breakouts above R1 or below S1 with 4h trend alignment and 1d volume confirmation capture
# high-probability moves. The 4h EMA50 provides trend filter to avoid counter-trend trades,
# while 1d volume surge ensures institutional participation. Session filter (08-20 UTC)
# reduces noise from low-liquidity periods. Designed for 15-30 trades/year to avoid fee drag.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d average volume for volume surge filter
    avg_volume_1d = pd.Series(df_1d['volume']).mean()
    
    # Calculate Camarilla pivot levels for previous day
    # Using previous day's OHLC to avoid look-ahead
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = prev_day_close + (prev_day_high - prev_day_low) * 1.1 / 12
    camarilla_s1 = prev_day_close - (prev_day_high - prev_day_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume surge: current 1h volume > 2x 1d average volume
    volume_surge = volume > (avg_volume_1d * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(50, 30)  # EMA50_4h and Camarilla calculation
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade during active hours
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 4h EMA50 direction
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Camarilla R1 + uptrend + volume surge
            if (close[i] > camarilla_r1_aligned[i]) and uptrend and volume_surge[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Camarilla S1 + downtrend + volume surge
            elif (close[i] < camarilla_s1_aligned[i]) and downtrend and volume_surge[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S1 or trend reverses
            if (close[i] < camarilla_s1_aligned[i]) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above Camarilla R1 or trend reverses
            if (close[i] > camarilla_r1_aligned[i]) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals