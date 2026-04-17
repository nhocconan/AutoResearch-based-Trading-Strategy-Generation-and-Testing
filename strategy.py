#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike filter and 1w EMA34 trend filter.
Long when price breaks above R3 with volume > 2.0x 1d average volume AND 1w EMA34 rising.
Short when price breaks below S3 with volume > 2.0x 1d average volume AND 1w EMA34 falling.
Exit when price touches the opposite Camarilla level (S3 for long, R3 for short).
Uses 1d for volume confirmation and Camarilla calculation, 1w for EMA34 trend filter.
Designed for low-frequency, high-conviction trades in both bull and bear markets.
Target: 12-25 trades/year per symbol (50-100 total over 4 years).
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
    
    # Get 1d data for volume MA and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_rising = ema_34_1w > np.roll(ema_34_1w, 1)
    ema_34_falling = ema_34_1w < np.roll(ema_34_1w, 1)
    ema_34_rising[0] = False
    ema_34_falling[0] = False
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_34_falling)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or
            np.isnan(ema_34_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 1d average volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
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
            
        # Camarilla levels (using R3/S3 for stronger breakouts)
        R3 = period_close + range_val * 1.1 / 4
        S3 = period_close - range_val * 1.1 / 4
        
        # Breakout conditions
        breakout_R3 = close[i] > R3
        breakout_S3 = close[i] < S3
        
        if position == 0:
            # Long: break above R3 with volume confirmation and rising 1w EMA
            if (breakout_R3 and volume_confirmed and ema_34_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume confirmation and falling 1w EMA
            elif (breakout_S3 and volume_confirmed and ema_34_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches S3
            if close[i] <= S3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches R3
            if close[i] >= R3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Volume_1d_1wEMA34_Trend"
timeframe = "12h"
leverage = 1.0