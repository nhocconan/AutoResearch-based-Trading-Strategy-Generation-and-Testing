#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: 12h Camarilla R1/S1 breakout with 1-week EMA50 trend filter and volume confirmation.
Long when price breaks above R1 with 1w uptrend and volume spike. Short when price breaks below S1 with 1w downtrend and volume spike.
Weekly EMA50 ensures trading with higher timeframe trend, reducing false signals in choppy markets.
Volume confirmation filters low-momentum breakouts. Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Weekly data for Camarilla calculation (based on prior week)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for current week using prior week's OHLC
    range_1w = high_1w - low_1w
    camarilla_r1 = close_1w + 0.275 * range_1w
    camarilla_s1 = close_1w - 0.275 * range_1w
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 week for proper timing)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50(1w) and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 + 1w uptrend + volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_50_1w_aligned[i]) and volume_spike[i]
            # Short: break below S1 + 1w downtrend + volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_50_1w_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price closes below R1 OR 1w trend turns down
            if (close[i] < camarilla_r1_aligned[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price closes above S1 OR 1w trend turns up
            if (close[i] > camarilla_s1_aligned[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0