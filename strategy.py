#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA200_TrendFilter_VolumeSpike
Hypothesis: Trade Camarilla R3/S3 breakouts with 1d EMA200 trend filter (stronger trend definition) and volume spike confirmation.
EMA200 provides robust long-term trend filter reducing whipsaws in bear markets. R3/S3 are stronger levels reducing false breakouts.
Only trade in direction of 1d trend to avoid counter-trend whipsaws. Discrete sizing 0.25 to manage risk and minimize fee churn.
Target: 20-40 trades/year to stay within fee drag limits.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA200 for trend filter (more robust than EMA50)
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_day_high - prev_day_low
    r3 = prev_day_close + 1.1 * camarilla_range / 4  # R3 level
    s3 = prev_day_close - 1.1 * camarilla_range / 4  # S3 level
    h3 = prev_day_close + 1.1 * camarilla_range / 6  # H3 level
    l3 = prev_day_close - 1.1 * camarilla_range / 6  # L3 level
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2.0x 20-period average (stricter filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for daily EMA200 (200) and volume MA (20)
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND daily trend bullish (close > EMA200) AND volume spike
            long_setup = (close[i] > r3_aligned[i]) and \
                         (close[i] > ema_200_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below S3 AND daily trend bearish (close < EMA200) AND volume spike
            short_setup = (close[i] < s3_aligned[i]) and \
                          (close[i] < ema_200_1d_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Camarilla H3/L3 range OR daily trend turns bearish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range OR daily trend turns bullish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA200_TrendFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0