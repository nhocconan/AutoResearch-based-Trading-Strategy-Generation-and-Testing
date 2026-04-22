#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R Extreme with 1w trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) with bullish 1w trend (price > EMA50) and volume spike.
Short when Williams %R > -20 (overbought) with bearish 1w trend (price < EMA50) and volume spike.
Exit when Williams %R returns to -50 (mean reversion).
Uses 1w EMA50 for trend filter to capture long-term trend and avoid whipsaws.
Designed for low trade frequency (7-25/year) to minimize fee drag.
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
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1d timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R lookback
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_avg_20[i])):
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
            # Long: Williams %R < -80 (oversold) with bullish 1w trend and volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema50_aligned[i] and  # Bullish trend: price above EMA50
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with bearish 1w trend and volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema50_aligned[i] and  # Bearish trend: price below EMA50
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to -50 (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns to -50 from below
                if williams_r[i] >= -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R returns to -50 from above
                if williams_r[i] <= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WilliamsR_Extreme_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0
#%%