#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla pivot breakout with 12-hour EMA trend filter and volume confirmation.
Goes long when price breaks above R1 with strong 12h EMA trend and volume spike.
Goes short when price breaks below S1 with strong 12h EMA trend and volume spike.
Exits when price returns to pivot point or trend weakens.
Uses 12h EMA for trend strength to avoid whipsaws in ranging markets.
Designed for low trade frequency (20-40/year) to minimize fee drag.
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
    
    # Load 12h data for EMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA (34-period)
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate daily Camarilla pivot levels
    # Need previous day's OHLC
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    # Calculate pivot and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R1 = close + (range_hl * 1.1 / 12)  # Using current close as base
    S1 = close - (range_hl * 1.1 / 12)
    R4 = close + (range_hl * 1.1 / 2)
    S4 = close - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to avoid issues with shifted data
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
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
            # Long: Price breaks above R1 with strong 12h EMA trend and volume
            if (close[i] > R1_aligned[i] and 
                close[i] > ema_12h_aligned[i] and  # Price above EMA = uptrend
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with strong 12h EMA trend and volume
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema_12h_aligned[i] and  # Price below EMA = downtrend
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to pivot OR price below EMA
                if close[i] < pivot_aligned[i] or close[i] < ema_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to pivot OR price above EMA
                if close[i] > pivot_aligned[i] or close[i] > ema_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0
#%%