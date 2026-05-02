#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13 to detect trend strength and exhaustion
# Bull Power = High - EMA13, Bear Power = Low - EMA13
# Strong Bull Power + price > EMA34 indicates sustained uptrend; strong Bear Power + price < EMA34 indicates downtrend
# Volume confirmation filters low-participation moves
# Works in both bull (riding trends) and bear (fading exhaustion) markets
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Discrete sizing 0.25 to balance profit potential and fee drag

name = "6h_ElderRay_Power_1dEMA34_Volume"
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
    
    # Calculate 1d Elder Ray Power components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for EMA13
        return np.zeros(n)
    
    # Previous day's data for Elder Ray (to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate EMA13 for previous day's close
    close_1d = prev_close  # Use shifted close for EMA calculation
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = prev_high - ema_13_1d
    bear_power = prev_low - ema_13_1d
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 1d EMA34 for trend filter
    close_1d_raw = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d_raw).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Strong Bull Power AND price > 1d EMA34 AND volume spike
            if (bull_power_aligned[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Strong Bear Power AND price < 1d EMA34 AND volume spike
            elif (bear_power_aligned[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power turns negative (trend weakening) OR price < EMA34 (trend change)
            if bull_power_aligned[i] <= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive (trend weakening) OR price > EMA34 (trend change)
            if bear_power_aligned[i] >= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals