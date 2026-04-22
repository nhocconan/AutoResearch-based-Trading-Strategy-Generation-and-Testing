#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian channel breakout (20-period) with 1w EMA(34) trend filter and volume confirmation.
Long when price breaks above Donchian upper with price above 1w EMA34 and volume > 1.5x average.
Short when price breaks below Donchian lower with price below 1w EMA34 and volume > 1.5x average.
Exit when price crosses Donchian middle or EMA trend flips.
Designed for low trade frequency (10-25/year) to minimize fee flood.
Works in bull (breakouts with trend) and bear (mean reversion via middle cross exits).
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
    
    # Calculate Donchian Channel (20-period) on 1d
    lookback = 20
    dc_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    dc_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    dc_middle = (dc_upper + dc_lower) / 2.0
    
    # Calculate 1w EMA(34)
    close_w = pd.Series(df_weekly['close'].values)
    ema_34_w = close_w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 1d timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_weekly, ema_34_w)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper with uptrend and volume
            if (close[i] > dc_upper[i] and 
                close[i] > ema_34_aligned[i] and  # Uptrend filter
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with downtrend and volume
            elif (close[i] < dc_lower[i] and 
                  close[i] < ema_34_aligned[i] and  # Downtrend filter
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle OR trend turns down
                if close[i] < dc_middle[i] or close[i] < ema_34_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle OR trend turns up
                if close[i] > dc_middle[i] or close[i] > ema_34_aligned[i]:
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