#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day VWAP filter and volume confirmation.
Long when price breaks above Donchian upper with price above daily VWAP and volume spike.
Short when price breaks below Donchian lower with price below daily VWAP and volume spike.
Exit when price crosses Donchian middle or price crosses daily VWAP in opposite direction.
Uses daily VWAP as trend filter to avoid whipsaws in ranging markets.
Designed for low trade frequency (15-35/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for VWAP filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Donchian Channel (20-period) on 4h
    lookback = 20
    dc_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    dc_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    dc_middle = (dc_upper + dc_lower) / 2.0
    
    # Calculate daily VWAP (typical price * volume cumulative)
    typical_price_d = (df_daily['high'] + df_daily['low'] + df_daily['close']) / 3.0
    vwap_d = (typical_price_d * df_daily['volume']).cumsum() / df_daily['volume'].cumsum()
    vwap_d_values = vwap_d.values
    
    # Align daily VWAP to 4h timeframe
    vwap_aligned = align_ltf_to_htf(prices, df_daily, vwap_d_values)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above Donchian upper with price above VWAP and volume
            if (close[i] > dc_upper[i] and 
                close[i] > vwap_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with price below VWAP and volume
            elif (close[i] < dc_lower[i] and 
                  close[i] < vwap_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle OR price crosses below VWAP
                if close[i] < dc_middle[i] or close[i] < vwap_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle OR price crosses above VWAP
                if close[i] > dc_middle[i] or close[i] > vwap_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_DonchianBreakout_1dVWAP_Volume"
timeframe = "4h"
leverage = 1.0
#%%