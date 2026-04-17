#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter.
Long when price breaks above Donchian upper band with volume > 1.5x 4h avg volume AND 1d EMA50 rising.
Short when price breaks below Donchian lower band with volume > 1.5x 4h avg volume AND 1d EMA50 falling.
Exit on opposite Donchian band touch or EMA trend reversal.
Uses 4h for execution and volume, 1d for EMA trend filter.
Designed to work in both bull and bear markets by following the 1d trend with volume confirmation.
Target: 20-50 trades/year per symbol (75-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_rising = ema_50_1d > np.roll(ema_50_1d, 1)
    ema_50_falling = ema_50_1d < np.roll(ema_50_1d, 1)
    ema_50_rising[0] = False
    ema_50_falling[0] = False
    
    # Align 1d EMA to 4h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper_band[i] = np.max(high[i-lookback+1:i+1])
        lower_band[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 4h volume MA (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(100, lookback)  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_upper = close[i] > upper_band[i]
        breakout_lower = close[i] < lower_band[i]
        
        # Exit conditions: touch opposite band or EMA trend reversal
        exit_long = close[i] < lower_band[i] or not ema_50_rising_aligned[i]
        exit_short = close[i] > upper_band[i] or not ema_50_falling_aligned[i]
        
        if position == 0:
            # Long: break above upper band with volume confirmation and rising 1d EMA
            if breakout_upper and volume_confirmed and ema_50_rising_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume confirmation and falling 1d EMA
            elif breakout_lower and volume_confirmed and ema_50_falling_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch lower band or EMA trend reversal
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch upper band or EMA trend reversal
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0