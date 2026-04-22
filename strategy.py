#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian(20) breakout with 1-week trend filter and volume confirmation.
Donchian breakouts capture breakout momentum. Weekly trend filter ensures we only trade
in the direction of the major trend, reducing counter-trend trades. Volume confirmation
filters out false breakouts. This should work in both bull and bear regimes by adapting
to the weekly trend. Target: 15-30 trades/year per symbol.
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
    
    # Load weekly data - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Determine weekly trend: price above/below EMA34
    bullish_trend = close_1w > ema_34_1w
    bearish_trend = close_1w < ema_34_1w
    
    # Align weekly trend to 12h timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_trend.astype(float))
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg_20[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i])):
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
            # Long: Donchian breakout above, bullish weekly trend, volume spike
            if (high[i] > donchian_high[i-1] and    # Break above upper band
                bullish_aligned[i] > 0.5 and        # Bullish weekly trend
                volume[i] > 2.0 * vol_avg_20[i]):   # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below, bearish weekly trend, volume spike
            elif (low[i] < donchian_low[i-1] and    # Break below lower band
                  bearish_aligned[i] > 0.5 and      # Bearish weekly trend
                  volume[i] > 2.0 * vol_avg_20[i]): # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle of Donchian channel or opposite breakout
            middle = (donchian_high[i] + donchian_low[i]) / 2
            exit_signal = False
            
            if position == 1:
                # Exit long: price drops below middle OR breaks below lower band
                if (close[i] < middle or low[i] < donchian_low[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises above middle OR breaks above upper band
                if (close[i] > middle or high[i] > donchian_high[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0