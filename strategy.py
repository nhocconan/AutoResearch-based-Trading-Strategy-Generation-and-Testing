#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation on 4h.
Long when price breaks above R1 AND 1d EMA34 uptrend AND volume > 1.5x 20-period average.
Short when price breaks below S1 AND 1d EMA34 downtrend AND volume > 1.5x 20-period average.
Uses discrete sizing (0.25) to minimize fee drag. Target: 75-200 trades over 4 years.
Works in bull/bear via 1d trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate pivot points from previous day
    # For 4h data, we need daily high/low/close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Align to 4h timeframe
    prev_high_4h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_4h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_4h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels: R1, S1
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    camarilla_range = prev_high_4h - prev_low_4h
    r1 = prev_close_4h + camarilla_range * 1.1 / 12
    s1 = prev_close_4h - camarilla_range * 1.1 / 12
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for volume MA, 1 for pivot)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R1 + 1d EMA34 uptrend + volume spike
        long_condition = (close[i] > r1[i] and 
                         ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                         volume_spike[i])
        
        # Short logic: price breaks below S1 + 1d EMA34 downtrend + volume spike
        short_condition = (close[i] < s1[i] and 
                          ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                          volume_spike[i])
        
        # Exit logic: opposite breakout or loss of trend
        exit_long = (close[i] < s1[i] or 
                    ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1])
        exit_short = (close[i] > r1[i] or 
                     ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0