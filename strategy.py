#!/usr/bin/env python3
# 6H_1W_1D_ElderRay_BullBearPower_Regime
# Hypothesis: On 6h timeframe, use weekly Elder Ray (bull/bear power) for trend direction and daily Elder Ray for entry timing.
# Bull power = high - EMA(13), Bear power = EMA(13) - low. Long when bull power rising and bear power negative (bullish momentum).
# Short when bear power rising and bull power negative (bearish momentum). Uses 13-period EMA for responsiveness.
# Includes volume confirmation to avoid false signals. Designed to work in both bull and bear markets by following momentum.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).

name = "6H_1W_1D_ElderRay_BullBearPower_Regime"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for trend direction (Elder Ray)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate EMA(13) for weekly close
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Weekly Elder Ray: Bull power = high - EMA13, Bear power = EMA13 - low
    bull_power_1w = high_1w - ema13_1w
    bear_power_1w = ema13_1w - low_1w
    
    # Weekly trend: rising bull power and falling bear power indicates bullish momentum
    bull_power_rising = bull_power_1w > np.roll(bull_power_1w, 1)
    bear_power_falling = bear_power_1w < np.roll(bear_power_1w, 1)
    bear_power_rising = bear_power_1w > np.roll(bear_power_1w, 1)
    bull_power_falling = bull_power_1w < np.roll(bull_power_1w, 1)
    
    # Handle first value for rolling comparison
    bull_power_rising[0] = False
    bear_power_falling[0] = False
    bear_power_rising[0] = False
    bull_power_falling[0] = False
    
    weekly_bullish = bull_power_rising & bear_power_falling
    weekly_bearish = bear_power_rising & bull_power_falling
    
    # Get daily data for entry timing (Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA(13) for daily close
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Daily Elder Ray
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Daily signals: look for momentum shifts
    bull_power_rising_1d = bull_power_1d > np.roll(bull_power_1d, 1)
    bear_power_rising_1d = bear_power_1d > np.roll(bear_power_1d, 1)
    
    # Handle first value
    bull_power_rising_1d[0] = False
    bear_power_rising_1d[0] = False
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.3)
    
    # Align weekly and daily indicators to 6h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    bull_power_rising_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_rising_1d.astype(float))
    bear_power_rising_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_rising_1d.astype(float))
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(bull_power_rising_1d_aligned[i]) or np.isnan(bear_power_rising_1d_aligned[i]) or
            np.isnan(volume_confirm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: weekly bullish momentum + daily bull power rising + volume confirmation
            if (weekly_bullish_aligned[i] == 1 and 
                bull_power_rising_1d_aligned[i] == 1 and 
                volume_confirm_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # Enter short: weekly bearish momentum + daily bear power rising + volume confirmation
            elif (weekly_bearish_aligned[i] == 1 and 
                  bear_power_rising_1d_aligned[i] == 1 and 
                  volume_confirm_aligned[i] == 1):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly momentum turns bearish or daily bear power rises
            if (weekly_bearish_aligned[i] == 1 or 
                bear_power_rising_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly momentum turns bullish or daily bull power rises
            if (weekly_bullish_aligned[i] == 1 or 
                bull_power_rising_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals