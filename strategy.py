#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h timeframe with 1d EMA34 trend filter and volume spike confirmation.
This strategy uses 12h primary timeframe (lower trade frequency) and 1d HTF for trend alignment.
Only trade in direction of 1d trend to avoid counter-trend whipsaws. Volume spike confirms institutional interest.
Discrete sizing 0.25 to manage risk and minimize fee churn.
Target: 12-30 trades/year to stay within fee drag limits for 12h timeframe.
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
    
    # Get daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_day_high - prev_day_low
    r1 = prev_day_close + 1.1 * camarilla_range / 12  # R1 level
    s1 = prev_day_close - 1.1 * camarilla_range / 12  # S1 level
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for daily EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND daily trend bullish (close > EMA34) AND volume spike
            long_setup = (close[i] > r1_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below S1 AND daily trend bearish (close < EMA34) AND volume spike
            short_setup = (close[i] < s1_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
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
            h1 = prev_day_close + 1.1 * camarilla_range / 6  # H1 level
            l1 = prev_day_close - 1.1 * camarilla_range / 6  # L1 level
            h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
            l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
            if (close[i] < h1_aligned[i] and close[i] > l1_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range OR daily trend turns bullish
            h1 = prev_day_close + 1.1 * camarilla_range / 6  # H1 level
            l1 = prev_day_close - 1.1 * camarilla_range / 6  # L1 level
            h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
            l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
            if (close[i] < h1_aligned[i] and close[i] > l1_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0