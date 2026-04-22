#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian(20) breakout with 1-week trend filter and volume confirmation.
Long when price breaks above Donchian high with bullish weekly trend and volume spike.
Short when price breaks below Donchian low with bearish weekly trend and volume spike.
Exit when price returns to Donchian midpoint.
Uses 1-week EMA20 for trend filter to capture long-term trend and avoid whipsaws.
Designed for low trade frequency (15-25/year) to minimize fee drag in bear markets.
"""
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
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align EMA20 to daily timeframe
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate daily Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2.0
    
    # Pre-calculate volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after lookback periods
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema20_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above Donchian high with bullish weekly trend and volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema20_aligned[i] and  # Bullish trend: price above weekly EMA20
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with bearish weekly trend and volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema20_aligned[i] and  # Bearish trend: price below weekly EMA20
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Donchian midpoint
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to midpoint
                if close[i] <= donchian_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to midpoint
                if close[i] >= donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_20_1wEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0
#%%