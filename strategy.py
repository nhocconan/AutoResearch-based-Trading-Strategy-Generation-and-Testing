#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_1dVolSpike
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1w EMA50 trend filter and 1d volume spike confirmation.
Primary timeframe 12h reduces trade frequency (target: 12-37 trades/year) to minimize fee drag.
1w EMA50 provides robust long-term trend filter to avoid counter-trend whipsaws in bear markets.
Only trade in direction of 1w trend. Volume spike confirms institutional participation.
Discrete sizing 0.25 to manage risk and minimize fee churn.
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels from previous day's OHLC (R1/S1 levels)
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_day_high - prev_day_low
    r1 = prev_day_close + 1.1 * camarilla_range / 12  # R1 level
    s1 = prev_day_close - 1.1 * camarilla_range / 12  # S1 level
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current 1d volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    # Align volume spike to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly EMA50 (50) and daily volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND weekly trend bullish (close > EMA50) AND volume spike
            long_setup = (close[i] > r1_aligned[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_spike_aligned[i]
            # Short: price breaks below S1 AND weekly trend bearish (close < EMA50) AND volume spike
            short_setup = (close[i] < s1_aligned[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_spike_aligned[i]
            
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
            # Exit: price re-enters Camarilla H3/L3 range OR weekly trend turns bearish
            # Calculate H3/L3 for exit
            h3 = prev_day_close + 1.1 * camarilla_range / 6
            l3 = prev_day_close - 1.1 * camarilla_range / 6
            h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
            l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range OR weekly trend turns bullish
            h3 = prev_day_close + 1.1 * camarilla_range / 6
            l3 = prev_day_close - 1.1 * camarilla_range / 6
            h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
            l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_1dVolSpike"
timeframe = "12h"
leverage = 1.0