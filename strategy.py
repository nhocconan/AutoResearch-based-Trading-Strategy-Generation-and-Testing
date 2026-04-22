#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d ATR-based volatility filter and volume confirmation.
Long when price breaks above Donchian upper with low volatility (ATR ratio < 1) and volume spike.
Short when price breaks below Donchian lower with low volatility and volume spike.
Exit when price crosses Donchian middle or volatility spikes (ATR ratio > 1.5).
Designed for low trade frequency (20-40/year) to minimize fee flood.
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
    
    # Load daily data for ATR filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate Donchian Channel (20-period) on 4h
    lookback = 20
    dc_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    dc_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    dc_middle = (dc_upper + dc_lower) / 2.0
    
    # Calculate 1d ATR (14-period)
    high_d = pd.Series(df_daily['high'].values)
    low_d = pd.Series(df_daily['low'].values)
    close_d = pd.Series(df_daily['close'].values)
    
    # True Range
    tr1 = high_d - low_d
    tr2 = abs(high_d - close_d.shift(1))
    tr3 = abs(low_d - close_d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_d = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate 14-day SMA of ATR for volatility regime
    atr_ma_d = pd.Series(atr_d).rolling(window=14, min_periods=14).mean()
    # ATR ratio: current ATR / 14-day average ATR
    atr_ratio_d = atr_d / atr_ma_d
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_daily, atr_ratio_d.values)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above Donchian upper with low volatility and volume spike
            if (close[i] > dc_upper[i] and 
                atr_ratio_aligned[i] < 1.0 and  # Low volatility regime
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with low volatility and volume spike
            elif (close[i] < dc_lower[i] and 
                  atr_ratio_aligned[i] < 1.0 and  # Low volatility regime
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle OR volatility spikes
                if close[i] < dc_middle[i] or atr_ratio_aligned[i] > 1.5:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle OR volatility spikes
                if close[i] > dc_middle[i] or atr_ratio_aligned[i] > 1.5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_DonchianBreakout_1dATRRatio_Volume"
timeframe = "4h"
leverage = 1.0
#%%