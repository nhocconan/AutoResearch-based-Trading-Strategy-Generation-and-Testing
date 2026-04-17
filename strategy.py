#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with 1w volume confirmation and 1d EMA50 trend filter.
Long when price breaks above R1 with volume > 1.5x 1w average volume AND 1d EMA50 rising.
Short when price breaks below S1 with volume > 1.5x 1w average volume AND 1d EMA50 falling.
Exit when price touches the opposite Camarilla level (S1 for long, R1 for short).
Uses 1w for volume confirmation and 1d for EMA50 trend filter and Camarilla calculation.
Designed to work in both bull and bear markets by following the 1d trend with volume confirmation.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for volume MA
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_rising = ema_50_1d > np.roll(ema_50_1d, 1)
    ema_50_falling = ema_50_1d < np.roll(ema_50_1d, 1)
    ema_50_rising[0] = False
    ema_50_falling[0] = False
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or
            np.isnan(ema_50_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 1w average volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Get the most recent completed 1d bar's OHLC for Camarilla
        high_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['high'].values)
        low_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['low'].values)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        
        period_high = high_1d_aligned[i]
        period_low = low_1d_aligned[i]
        period_close = close_1d_aligned[i]
        
        range_val = period_high - period_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        R1 = period_close + range_val * 1.1 / 12
        S1 = period_close - range_val * 1.1 / 12
        
        # Breakout conditions
        breakout_R1 = close[i] > R1
        breakout_S1 = close[i] < S1
        
        if position == 0:
            # Long: break above R1 with volume confirmation and rising 1d EMA
            if (breakout_R1 and volume_confirmed and ema_50_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation and falling 1d EMA
            elif (breakout_S1 and volume_confirmed and ema_50_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches S1
            if close[i] <= S1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches R1
            if close[i] >= R1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Volume_1w_1dEMA50_Trend"
timeframe = "1d"
leverage = 1.0