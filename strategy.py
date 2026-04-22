#!/usr/bin/env python3

"""
Hypothesis: 6-hour Williams %R with 1-day trend filter and volume spike confirmation.
Williams %R identifies overbought/oversold conditions. Using daily trend filter avoids
counter-trend trades. Volume spikes confirm institutional interest at reversals.
This should work in both bull and bear regimes by adapting to the daily trend.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_r(high, low, close, period=14):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily trend using EMA crossover (fast/slow)
    close_1d = df_1d['close'].values
    ema_fast = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    bullish_trend = ema_fast > ema_slow
    bearish_trend = ema_fast < ema_slow
    
    # Align daily trend to 6h
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Calculate 6h Williams %R
    wr = williams_r(high, low, close, 14)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(wr[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Williams %R oversold (< -80), bullish daily trend, volume spike
            if (wr[i] < -80 and 
                bullish_aligned[i] > 0.5 and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), bearish daily trend, volume spike
            elif (wr[i] > -20 and 
                  bearish_aligned[i] > 0.5 and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to neutral range or opposite extreme
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 (momentum fading)
                if wr[i] > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R falls below -50 (momentum fading)
                if wr[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0