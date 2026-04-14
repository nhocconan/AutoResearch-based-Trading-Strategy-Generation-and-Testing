#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d EMA200 filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. 
# Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and increasing, Bear Power < 0 and decreasing, price above 1d EMA200
# Short when Bear Power < 0 and decreasing, Bull Power < 0 and decreasing, price below 1d EMA200
# Volume confirmation filters weak breakouts
# Works in bull/bear by measuring institutional buying/selling pressure
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Smooth the power signals to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Get 1d EMA200 for trend filter (using mtf_data)
    df_1d = get_htf_data(prices, '1d')
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: volume > 1.5x average volume (30-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=30, min_periods=30).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volume filter
        if vol < 1.5 * avg_vol[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull power positive and rising, bear power negative, price above 1d EMA200
            if (bull_power_smooth[i] > 0 and 
                bull_power_smooth[i] > bull_power_smooth[i-1] and
                bear_power_smooth[i] < 0 and
                price > ema200_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: Bear power negative and falling, bull power negative, price below 1d EMA200
            elif (bear_power_smooth[i] < 0 and 
                  bear_power_smooth[i] < bear_power_smooth[i-1] and
                  bull_power_smooth[i] < 0 and
                  price < ema200_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull power turns negative or price crosses below 1d EMA200
            if bull_power_smooth[i] <= 0 or price < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear power turns positive or price crosses above 1d EMA200
            if bear_power_smooth[i] >= 0 or price > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_1dEMA200_VolumeFilter"
timeframe = "6h"
leverage = 1.0