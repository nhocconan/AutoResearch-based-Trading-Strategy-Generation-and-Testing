#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 12h EMA Trend Filter and 1d Volume Spike
# Uses Elder Ray Bull Power (High - EMA13) and Bear Power (EMA13 - Low) to measure bull/bear strength
# Long when Bull Power > 0, Bear Power rising, price > 12h EMA200, and 1d volume spike
# Short when Bear Power < 0, Bull Power falling, price < 12h EMA200, and 1d volume spike
# Designed to capture institutional buying/selling pressure with trend and volume confirmation
# Target: 60-120 total trades over 4 years (15-30/year)

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)  # Bull Power from 12h
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)  # Bear Power from 12h
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200  # for EMA200 calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_12h_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: Bull Power positive, Bear Power rising, price above 12h EMA200, volume spike
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] > bear_power_aligned[i-1] and  # Bear Power rising (less negative)
                price > ema200_12h_aligned[i] and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):       # Volume spike
                position = 1
                signals[i] = position_size
            # Short setup: Bear Power negative, Bull Power falling, price below 12h EMA200, volume spike
            elif (bear_power_aligned[i] < 0 and 
                  bull_power_aligned[i] < bull_power_aligned[i-1] and  # Bull Power falling (less positive)
                  price < ema200_12h_aligned[i] and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):       # Volume spike
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative or volume drops
            if bull_power_aligned[i] <= 0 or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power turns positive or volume drops
            if bear_power_aligned[i] >= 0 or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_12hEMA200_1dVolume"
timeframe = "6h"
leverage = 1.0