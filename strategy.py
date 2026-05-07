#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla R3/S3 breakouts on 6h capture institutional-level breakouts with follow-through.
# Confirmed by 1d EMA34 trend filter and volume spike (>1.8x average).
# Works in bull markets via long breakouts and bear via short breakdowns.
# Volume filter reduces false breakouts, trend filter avoids counter-trend trades.
# Target: 10-30 trades per year (~40-120 over 4 years) with position size 0.25.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Need previous day's OHLC - use 1d data shifted
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Align previous day's OHLC to 6t timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla calculations
    range_prev = prev_high_aligned - prev_low_aligned
    # Avoid division by zero
    range_prev = np.where(range_prev == 0, 1e-10, range_prev)
    
    # Camarilla R3 and S3 levels
    r3 = prev_close_aligned + (range_prev * 1.1 / 4)
    s3 = prev_close_aligned - (range_prev * 1.1 / 4)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above R3 or below S3
        breakout_up = close[i] > r3[i]
        breakout_down = close[i] < s3[i]
        
        # Volume confirmation: volume > 1.8x average
        volume_confirm = vol_ratio[i] > 1.8
        
        # Trend filter from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks back below S3 or trend reversal
            if close[i] < s3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks back above R3 or trend reversal
            if close[i] > r3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals