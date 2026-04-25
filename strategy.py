#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike_DynamicExit
Hypothesis: Trade Camarilla H3/L3 breakouts with 1d EMA50 trend filter and volume spike confirmation.
Dynamic exit: close position when price crosses the opposite Camarilla level (H3 for longs, L3 for shorts) 
or when trend reverses. Uses discrete position sizing 0.25 to limit fee churn and manage drawdown.
Designed to work in both bull and bear markets by only trading with the daily trend.
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
    
    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_day_high - prev_day_low
    h3 = prev_day_close + 1.1 * camarilla_range / 6  # H3 level
    l3 = prev_day_close - 1.1 * camarilla_range / 6  # L3 level
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for daily EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND daily trend bullish (close > EMA50) AND volume spike
            long_setup = (close[i] > h3_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below L3 AND daily trend bearish (close < EMA50) AND volume spike
            short_setup = (close[i] < l3_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
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
            # Exit: price crosses below L3 (opposite level) OR daily trend turns bearish
            if (close[i] < l3_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price crosses above H3 (opposite level) OR daily trend turns bullish
            if (close[i] > h3_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike_DynamicExit"
timeframe = "4h"
leverage = 1.0