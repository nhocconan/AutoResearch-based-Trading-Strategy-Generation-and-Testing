#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-day EMA34 trend filter and volume confirmation.
# Elder Ray measures bullish/bearish power relative to EMA13: BullPower = High - EMA13, BearPower = Low - EMA13.
# Long when BullPower > 0 and rising, BearPower < 0 and falling (bullish momentum).
# Short when BearPower < 0 and falling, BullPower > 0 and rising (bearish momentum).
# Uses 1-day EMA34 for trend filter to align with higher timeframe direction.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets (captures sustained uptrends) and bear markets (captures sustained downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Slope of Bull Power and Bear Power (1-period change)
    bull_power_slope = np.diff(bull_power, prepend=bull_power[0])
    bear_power_slope = np.diff(bear_power, prepend=bear_power[0])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: Bull Power positive and rising, Bear Power negative and falling, uptrend, volume
        if (bull_power[i] > 0 and bull_power_slope[i] > 0 and 
            bear_power[i] < 0 and bear_power_slope[i] < 0 and
            close[i] > ema34_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: Bear Power negative and falling, Bull Power positive and rising, downtrend, volume
        elif (bear_power[i] < 0 and bear_power_slope[i] < 0 and
              bull_power[i] > 0 and bull_power_slope[i] > 0 and
              close[i] < ema34_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or loss of momentum
        elif position == 1 and (close[i] <= ema34_1d_aligned[i] or bull_power[i] <= 0):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] >= ema34_1d_aligned[i] or bear_power[i] >= 0):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0