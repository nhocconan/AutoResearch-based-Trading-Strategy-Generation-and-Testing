#!/usr/bin/env python3
"""
12h Trading Range Breakout with Volume Confirmation and Trend Filter
Long when price breaks above 12h high with volume spike and bullish daily trend.
Short when price breaks below 12h low with volume spike and bearish daily trend.
Exit when price returns to 12h midpoint or trend reverses.
Designed for low trade frequency (15-30/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_d = pd.Series(df_daily['close'].values)
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_daily, ema34_d)
    
    # Calculate 12h high and low from previous completed 12h candles
    # Use rolling window of 2 periods (current and previous) to get previous 12h high/low
    high_12h = pd.Series(high).rolling(window=2, min_periods=2).max().shift(1).values
    low_12h = pd.Series(low).rolling(window=2, min_periods=2).min().shift(1).values
    mid_12h = (high_12h + low_12h) / 2.0
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume lookback
        # Skip if data not ready
        if (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above 12h high with volume spike and bullish daily trend
            if (close[i] > high_12h[i] and 
                close[i] > ema34_aligned[i] and  # Bullish trend: price above daily EMA34
                volume[i] > 1.8 * vol_avg_20[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h low with volume spike and bearish daily trend
            elif (close[i] < low_12h[i] and 
                  close[i] < ema34_aligned[i] and  # Bearish trend: price below daily EMA34
                  volume[i] > 1.8 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to 12h midpoint OR trend turns bearish
                if close[i] <= mid_12h[i] or close[i] < ema34_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to 12h midpoint OR trend turns bullish
                if close[i] >= mid_12h[i] or close[i] > ema34_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_RangeBreakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0