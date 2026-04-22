#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian channel breakout with 1w EMA trend filter and volume confirmation.
Long when price breaks above Donchian upper (20) with price above weekly EMA(34) and volume > 1.5x average.
Short when price breaks below Donchian lower (20) with price below weekly EMA(34) and volume > 1.5x average.
Exit when price crosses Donchian middle.
Designed for low trade frequency (10-25/year) to minimize fee damp on daily timeframe.
Works in bull (breakouts) and bear (breakdowns) with trend filter preventing counter-trend whipsaws.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for EMA filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 35:
        return np.zeros(n)
    
    # Calculate Donchian Channel (20-period) on daily
    lookback = 20
    dc_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    dc_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    dc_middle = (dc_upper + dc_lower) / 2.0
    
    # Calculate weekly EMA(34)
    weekly_close = pd.Series(df_weekly['close'].values)
    weekly_ema = weekly_close.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper with weekly uptrend and volume
            if (close[i] > dc_upper[i] and 
                close[i] > weekly_ema_aligned[i] and  # Above weekly EMA (uptrend)
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with weekly downtrend and volume
            elif (close[i] < dc_lower[i] and 
                  close[i] < weekly_ema_aligned[i] and  # Below weekly EMA (downtrend)
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price crosses Donchian middle
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

name = "1D_DonchianBreakout_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0
#%%