#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter and Donchian channel - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    # Calculate 12h EMA25 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema25_12h = close_12h.ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Align EMA25 to 6h timeframe
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Highest high of last 20 periods
    high_max_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    low_min_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, high_max_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, low_min_20)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after EMA lookback
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema25_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with bullish 12h trend and volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema25_12h_aligned[i] and  # Bullish trend: price above EMA25
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with bearish 12h trend and volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema25_12h_aligned[i] and  # Bearish trend: price below EMA25
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Donchian band
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to or below Donchian low
                if close[i] <= donchian_low_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to or above Donchian high
                if close[i] >= donchian_high_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_20_12hEMA25_Trend_Volume"
timeframe = "6h"
leverage = 1.0
#%%