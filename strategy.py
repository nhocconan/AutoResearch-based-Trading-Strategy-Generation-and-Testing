#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Enter long when Bull Power > 0 and rising (current > previous) and close > 1d EMA34 and volume > 1.5x 20-bar average.
# Enter short when Bear Power < 0 and falling (current < previous) and close < 1d EMA34 and volume > 1.5x 20-bar average.
# Exit when power crosses zero (Bull Power <= 0 for long, Bear Power >= 0 for short).
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Elder Ray measures buying/selling pressure relative to EMA13, effective in both bull and bear markets.
# 1d EMA34 filter ensures trades align with higher timeframe trend, reducing whipsaws.
# Volume confirmation adds conviction to breakouts.

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and EMA34 (MTF structure/trend)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Bull Power and Bear Power
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d  # Buying pressure
    bear_power = low_1d - ema_13_1d   # Selling pressure (negative values indicate pressure)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d EMA34 bias
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        # Elder Ray conditions with momentum (rising/falling power)
        bull_power_current = bull_power_aligned[i]
        bull_power_previous = bull_power_aligned[i-1]
        bear_power_current = bear_power_aligned[i]
        bear_power_previous = bear_power_aligned[i-1]
        
        bull_power_rising = bull_power_current > bull_power_previous
        bear_power_falling = bear_power_current < bear_power_previous
        
        # Entry conditions
        long_entry = (bull_power_current > 0) and bull_power_rising and bullish_bias and vol_confirm
        short_entry = (bear_power_current < 0) and bear_power_falling and bearish_bias and vol_confirm
        
        # Exit conditions: power crosses zero
        long_exit = bull_power_current <= 0
        short_exit = bear_power_current >= 0
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals