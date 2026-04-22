#!/usr/bin/env python3

"""
Hypothesis: 1-day Donchian channel breakout with 1-week trend filter and volume confirmation.
The Donchian channel identifies breakouts from price ranges.
The 1-week trend filter ensures trades align with the weekly trend to avoid counter-trend trades.
Volume spikes confirm institutional participation at breakout points.
This strategy aims to capture strong momentum moves in both bull and bear markets by
trading breakouts of the Donchian channel with trend and volume confirmation.
Target: 7-25 trades/year per symbol (30-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, period=20):
    """Calculate Donchian channel: upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max()
    lower = pd.Series(low).rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d Donchian data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Donchian channel (20-period)
    upper_1d, lower_1d = calculate_donchian(
        df_1d['high'].values, df_1d['low'].values
    )
    
    # Align Donchian to 1d timeframe (already aligned, but for safety)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA for trend filter (20-period)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (00-23 UTC for daily)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # No session filter for daily timeframe - trade all hours
        
        if position == 0:
            # Long: price breaks above upper band, above weekly EMA, volume spike
            if (close[i] > upper_1d_aligned[i] and                    # Price above upper band
                close[i] > ema_20_1w_aligned[i] and                  # Above weekly EMA (bullish trend)
                volume[i] > 2.5 * vol_avg_20[i]):                    # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band, below weekly EMA, volume spike
            elif (close[i] < lower_1d_aligned[i] and                 # Price below lower band
                  close[i] < ema_20_1w_aligned[i] and                # Below weekly EMA (bearish trend)
                  volume[i] > 2.5 * vol_avg_20[i]):                  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite side of band or crosses weekly EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below lower band or below weekly EMA
                if close[i] < lower_1d_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above upper band or above weekly EMA
                if close[i] > upper_1d_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_20_1wEMA20_Volume"
timeframe = "1d"
leverage = 1.0