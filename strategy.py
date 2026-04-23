#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1w EMA(50) trend filter + volume confirmation (>1.5x 20-period average)
- Elder Ray measures bull/bear power via EMA(13): Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
- Long when Bull Power > 0 and rising, Bear Power < 0 and falling (strong bullish momentum)
- Short when Bear Power < 0 and falling, Bull Power > 0 and rising (strong bearish momentum)
- 1w EMA(50) ensures trades align with weekly trend to avoid counter-trend whipsaws
- Volume confirmation validates breakout strength
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with weekly trend
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
    
    # Get 1w data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Elder Ray components (Bull/Bear Power) using EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA(13)
    bear_power = low - ema_13   # Bear Power: Low - EMA(13)
    
    # Calculate smoothed Bull/Bear Power for trend confirmation
    bull_power_smooth = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13)  # EMA(50) 1w, EMA(13)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bull_power_smooth[i]) or
            np.isnan(bear_power_smooth[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray conditions with momentum
        # Long: Bull Power rising and positive, Bear Power falling and negative
        bull_rising = bull_power_smooth[i] > bull_power_smooth[i-1]
        bull_positive = bull_power_smooth[i] > 0
        bear_falling = bear_power_smooth[i] < bear_power_smooth[i-1]
        bear_negative = bear_power_smooth[i] < 0
        
        long_signal = bull_rising and bull_positive and bear_falling and bear_negative
        
        # Short: Bear Power falling and negative, Bull Power rising and positive
        bear_rising = bear_power_smooth[i] > bear_power_smooth[i-1]  # Actually rising (less negative)
        bull_falling = bull_power_smooth[i] < bull_power_smooth[i-1]  # Actually falling (less positive)
        
        short_signal = bear_rising and bear_negative and bull_falling and bull_positive
        
        if position == 0:
            # Entry conditions: Elder Ray signal + weekly trend + volume confirmation
            if (long_signal and 
                close[i] > ema_50_1w_aligned[i] and  # Weekly uptrend
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            elif (short_signal and 
                  close[i] < ema_50_1w_aligned[i] and  # Weekly downtrend
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Elder Ray signal reversal or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Elder Ray turns bearish or weekly trend turns down
                if (not long_signal or close[i] < ema_50_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Elder Ray turns bullish or weekly trend turns up
                if (not short_signal or close[i] > ema_50_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1wEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0