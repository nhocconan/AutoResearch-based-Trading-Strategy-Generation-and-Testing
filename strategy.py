#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: Trade 1h Camarilla R1/S1 breakouts with 4h EMA34 trend filter and volume confirmation (>1.8x 20-bar MA). 
Uses 4h/1d for signal direction, 1h only for entry timing. Session filter (08-20 UTC) reduces noise. 
Discrete sizing 0.20 minimizes fee drag. Target: 60-150 total trades over 4 years (15-37/year).
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for Camarilla calculation (structure)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar's OHLC
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12   # R1 level
    s1 = prev_close_1d - 1.1 * camarilla_range / 12   # S1 level
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 4h EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           np.isnan(ema_34_4h_aligned[i]) or \
           np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND 4h trend bullish (close > EMA34) AND volume confirm
            long_setup = (close[i] > r1_aligned[i]) and \
                         (close[i] > ema_34_4h_aligned[i]) and \
                         volume_confirm[i]
            # Short: price breaks below S1 AND 4h trend bearish (close < EMA34) AND volume confirm
            short_setup = (close[i] < s1_aligned[i]) and \
                          (close[i] < ema_34_4h_aligned[i]) and \
                          volume_confirm[i]
            
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
            # Exit: price re-enters Camarilla R1/S1 range OR 4h trend turns bearish
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or \
               (close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price re-enters Camarilla R1/S1 range OR 4h trend turns bullish
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or \
               (close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0