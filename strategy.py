#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # 12h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Daily SMA200
    sma200_daily = pd.Series(close_daily).rolling(window=200, min_periods=200).mean().values
    sma200_aligned = align_htf_to_ltf(prices, df_daily, sma200_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% of capital
    
    for i in range(n):
        # Skip if any essential value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(sma200_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian upper with volume confirmation and price above daily SMA200
            if (close[i] > donchian_upper[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > sma200_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian lower with volume confirmation and price below daily SMA200
            elif (close[i] < donchian_lower[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < sma200_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below Donchian middle
            if close[i] < donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price above Donchian middle
            if close[i] > donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_SMA200_Volume"
timeframe = "12h"
leverage = 1.0