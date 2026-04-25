#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1w EMA50 trend filter and volume confirmation (>2.0x 30-bar MA).
12h timeframe targets 12-37 trades/year to minimize fee drag. Camarilla R1/S1 provides strong support/resistance.
1w EMA50 filter ensures trading with higher timeframe trend (bull/bear adaptation). Volume confirmation adds conviction.
Discrete sizing 0.25 balances profit and fee drag. Works in bull/bear: trend filter adapts, volume confirms validity.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla calculation (using previous 1d bar's OHLC)
    df_1d = get_htf_data(prices, '1d')
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12   # R1 level
    s1 = prev_close_1d - 1.1 * camarilla_range / 12   # S1 level
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1w EMA50 (50) and volume MA (30)
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND 1w trend bullish (close > EMA50) AND volume confirm
            long_setup = (close[i] > r1_aligned[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_confirm[i]
            # Short: price breaks below S1 AND 1w trend bearish (close < EMA50) AND volume confirm
            short_setup = (close[i] < s1_aligned[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_confirm[i]
            
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
            # Exit: price re-enters Camarilla R1/S1 range OR 1w trend turns bearish
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla R1/S1 range OR 1w trend turns bullish
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0