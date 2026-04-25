#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike
Hypothesis: Trade 6h Camarilla R3/S3 breakouts with 12h EMA34 trend filter and volume spike confirmation.
12h EMA34 provides medium-term trend filter reducing whipsaws in choppy markets. R3/S3 are stronger levels.
Only trade in direction of 12h trend to avoid counter-trend whipsaws. Discrete sizing 0.25 to manage fee drag.
Target: 12-30 trades/year (~50-120 over 4 years) to stay within 6h fee drag limits.
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
    
    # Get 12h data for trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from previous 12h bar's OHLC
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    prev_close_12h = df_12h['close'].shift(1).values
    
    camarilla_range = prev_high_12h - prev_low_12h
    r3 = prev_close_12h + 1.1 * camarilla_range / 4  # R3 level
    s3 = prev_close_12h - 1.1 * camarilla_range / 4  # S3 level
    h3 = prev_close_12h + 1.1 * camarilla_range / 6  # H3 level
    l3 = prev_close_12h - 1.1 * camarilla_range / 6  # L3 level
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 12h EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND 12h trend bullish (close > EMA34) AND volume spike
            long_setup = (close[i] > r3_aligned[i]) and \
                         (close[i] > ema_34_12h_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below S3 AND 12h trend bearish (close < EMA34) AND volume spike
            short_setup = (close[i] < s3_aligned[i]) and \
                          (close[i] < ema_34_12h_aligned[i]) and \
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
            # Exit: price re-enters Camarilla H3/L3 range OR 12h trend turns bearish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3/L3 range OR 12h trend turns bullish
            if (close[i] < h3_aligned[i] and close[i] > l3_aligned[i]) or \
               (close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0