#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 12-hour EMA and 1-day volume filter
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 and Bear Power < 0 and volume > 1.5x 20-period average
# Short when Bull Power < 0 and Bear Power > 0 and volume > 1.5x 20-period average
# Exit when Bull Power and Bear Power have same sign (both positive or both negative)
# Position size: 0.25
# Uses 12-hour EMA for power calculation and 1-day volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_elder_ray_12h_ema_1d_vol_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour EMA (13-period)
    close_12h = df_12h['close'].values
    ema_13 = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_12h, ema_13)
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema_13_aligned
    bear_power = ema_13_aligned - low
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Bull Power and Bear Power both positive (bullish exhaustion)
            if bull_power[i] > 0 and bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power and Bear Power both negative (bearish exhaustion)
            if bull_power[i] < 0 and bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            
            # Long: Bull Power > 0 and Bear Power < 0 (strong bullish momentum)
            if bull_power[i] > 0 and bear_power[i] < 0 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 and Bear Power > 0 (strong bearish momentum)
            elif bull_power[i] < 0 and bear_power[i] > 0 and volume_filter:
                signals[i] = -0.25
                position = -1
    
    return signals