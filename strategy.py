#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d EMA200 filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 AND price > 1d EMA200 AND volume > 1.5x 20-period average.
# Short when Bear Power < 0 AND price < 1d EMA200 AND volume > 1.5x 20-period average.
# Exit when power crosses zero or price crosses 1d EMA200.
# Uses EMA13 for power calculation, EMA200 for trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency.

name = "6h_ElderRay_1dEMA200_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for EMA200 and volume filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # EMA13 for Elder Ray (6h timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Daily EMA200
    ema200_d = pd.Series(df_d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200 = align_htf_to_ltf(prices, df_d, ema200_d)
    
    # Daily volume filter: current volume > 1.5x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.5 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 to be valid
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema200[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, price above 1d EMA200, high volume
            long_cond = (bull_power[i] > 0) and (close[i] > ema200[i]) and volume_filter[i]
            # Short conditions: Bear Power < 0, price below 1d EMA200, high volume
            short_cond = (bear_power[i] < 0) and (close[i] < ema200[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR price crosses below 1d EMA200
            if (bull_power[i] <= 0) or (close[i] < ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 OR price crosses above 1d EMA200
            if (bear_power[i] >= 0) or (close[i] > ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals