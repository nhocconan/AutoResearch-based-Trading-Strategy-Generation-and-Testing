#!/usr/bin/env python3

"""
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
The Donchian channel identifies breakout points with clear support/resistance.
The weekly EMA50 ensures trades align with the higher timeframe trend to avoid counter-trend trades.
Volume spikes confirm institutional participation at breakout points.
This strategy aims to capture strong momentum moves in both bull and bear markets by
trading breakouts of the daily Donchian channel with weekly trend and volume confirmation.
Target: 7-25 trades/year per symbol (30-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) - upper and lower bands
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Load weekly data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: price breaks above Donchian upper, above weekly EMA50, volume spike
            if (close[i] > donchian_upper_aligned[i] and                    # Price above upper band
                close[i] > ema_50_1w_aligned[i] and                        # Above weekly EMA50 (bullish trend)
                volume[i] > 2.0 * vol_avg_20[i]):                          # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, below weekly EMA50, volume spike
            elif (close[i] < donchian_lower_aligned[i] and                 # Price below lower band
                  close[i] < ema_50_1w_aligned[i] and                      # Below weekly EMA50 (bearish trend)
                  volume[i] > 2.0 * vol_avg_20[i]):                        # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite side of Donchian band or crosses weekly EMA50
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below lower band or below weekly EMA50
                if close[i] < donchian_lower_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above upper band or above weekly EMA50
                if close[i] > donchian_upper_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0