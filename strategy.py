#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R mean reversion with 1-week trend filter and volume confirmation.
In overbought conditions (%R > -20) with bearish weekly trend, short.
In oversold conditions (%R < -80) with bullish weekly trend, long.
Volume surge confirms the reversal signal.
Designed for low trade frequency (12-37/year) to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 12h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate weekly EMA12 trend
    ema12_weekly = pd.Series(df_weekly['close'].values).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema12_aligned = align_htf_to_ltf(prices, df_weekly, ema12_weekly)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema12_aligned[i]) or 
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
            # Long: Oversold (%R < -80) with bullish weekly trend and volume
            if (williams_r[i] < -80 and 
                close[i] > ema12_aligned[i] and  # Price above weekly EMA = bullish trend
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Overbought (%R > -20) with bearish weekly trend and volume
            elif (williams_r[i] > -20 and 
                  close[i] < ema12_aligned[i] and  # Price below weekly EMA = bearish trend
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral zone (-50)
            if position == 1:
                if williams_r[i] > -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r[i] < -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_WilliamsR_1wEMA12Trend_Volume"
timeframe = "12h"
leverage = 1.0
#%%