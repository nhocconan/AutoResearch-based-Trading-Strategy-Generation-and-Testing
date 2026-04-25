#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrendFilter_VolumeSpike
Hypothesis: Trade 1h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and volume spike confirmation.
Uses 4h trend for signal direction (reduces whipsaws), 1h for precise entry timing, volume spike for confirmation.
Session filter (08-20 UTC) reduces noise trades. Discrete sizing 0.20 to minimize fee churn.
Target: 15-35 trades/year to stay within 1h fee drag limits.
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
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get daily data for Camarilla levels (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_day_high - prev_day_low
    r1 = prev_day_close + 1.1 * camarilla_range / 12  # R1 level
    s1 = prev_day_close - 1.1 * camarilla_range / 12  # S1 level
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 1.8x 24-period average (1h = 24h lookback)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma_24)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for daily Camarilla (1), 4h EMA50 (50), volume MA (24)
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND 4h trend bullish (close > EMA50) AND volume spike
            long_setup = (close[i] > r1_aligned[i]) and \
                         (close[i] > ema_50_4h_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below S1 AND 4h trend bearish (close < EMA50) AND volume spike
            short_setup = (close[i] < s1_aligned[i]) and \
                          (close[i] < ema_50_4h_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price re-enters Camarilla range OR 4h trend turns bearish
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or \
               (close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price re-enters Camarilla range OR 4h trend turns bullish
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or \
               (close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrendFilter_VolumeSpike"
timeframe = "1h"
leverage = 1.0