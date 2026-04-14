#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with volume spike
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, with volume spike
# Uses 1d EMA200 as trend filter: only long when price > EMA200, short when price < EMA200
# Volume confirmation: current 1d volume > 1.5x 20-period average
# Designed to capture institutional buying/selling pressure with trend and volume filters
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 13-period EMA for Elder Ray (using 1d close)
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power (1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Calculate 200-period EMA for trend filter (1d)
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 20-period volume average (1d)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for EMA200 and other calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: Bull Power positive and rising, Bear Power negative, price above EMA200, volume spike
            if (bull_power_aligned[i] > 0 and 
                i > start and bull_power_aligned[i] > bull_power_aligned[i-1] and  # Bull Power rising
                bear_power_aligned[i] < 0 and
                price > ema200_aligned[i] and
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: Bear Power negative and falling, Bull Power positive, price below EMA200, volume spike
            elif (bear_power_aligned[i] < 0 and 
                  i > start and bear_power_aligned[i] < bear_power_aligned[i-1] and  # Bear Power falling
                  bull_power_aligned[i] > 0 and
                  price < ema200_aligned[i] and
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative or price crosses below EMA200
            if bull_power_aligned[i] <= 0 or price < ema200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power turns positive or price crosses above EMA200
            if bear_power_aligned[i] >= 0 or price > ema200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_EMA200_Volume"
timeframe = "6h"
leverage = 1.0