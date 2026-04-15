#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w trend filter, Donchian channel breakout, volume confirmation
# Weekly trend filter reduces false breakouts in sideways markets
# Volume confirmation ensures breakouts have conviction
# Designed to work in both bull (breakouts continue) and bear (false breakouts filtered) markets
# Target: 15-25 trades/year to stay under fee drag limits

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Daily Donchian channel (20-period)
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_donchian = high_rolling.values
    lower_donchian = low_rolling.values
    
    # Volume confirmation: current > 1.8x median of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.8 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema_aligned[i]) or np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above weekly EMA AND above Donchian upper + volume
        if (close[i] > weekly_ema_aligned[i] and 
            close[i] > upper_donchian[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below weekly EMA AND below Donchian lower + volume
        elif (close[i] < weekly_ema_aligned[i] and 
              close[i] < lower_donchian[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back below/above weekly EMA (trend change)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < weekly_ema_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > weekly_ema_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyEMA_Donchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0