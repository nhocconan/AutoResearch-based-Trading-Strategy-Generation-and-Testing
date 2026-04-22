#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w trend filter.
Long when price breaks above Donchian upper with strong weekly trend and above-average volume.
Short when price breaks below Donchian lower with strong weekly trend and above-average volume.
Exit when price crosses Donchian middle.
Designed for low trade frequency (20-40/year) to minimize fee drift and improve generalization.
Works in both bull and bear markets via weekly trend filter that avoids counter-trend trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate Donchian Channel (20-period) on 4h
    lookback = 20
    dc_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    dc_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    dc_middle = (dc_upper + dc_lower) / 2.0
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_weekly = pd.Series(df_weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Calculate 1d volume average (20-period) for confirmation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    vol_avg_20_daily = pd.Series(df_daily['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_aligned[i])):
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
            # Long: Price breaks above Donchian upper with weekly uptrend and volume
            if (close[i] > dc_upper[i] and 
                close[i] > ema_50_aligned[i] and  # Weekly uptrend
                volume[i] > vol_avg_aligned[i]):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with weekly downtrend and volume
            elif (close[i] < dc_lower[i] and 
                  close[i] < ema_50_aligned[i] and  # Weekly downtrend
                  volume[i] > vol_avg_aligned[i]):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle
                if close[i] < dc_middle[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle
                if close[i] > dc_middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_DonchianBreakout_1wEMA50_1dVolume"
timeframe = "4h"
leverage = 1.0
#%%