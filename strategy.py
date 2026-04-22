#!/usr/bin/env python3

"""
Hypothesis: 6-hour Williams %R mean reversion with 1-day trend filter and volume confirmation.
Williams %R identifies overbought/oversold conditions. The 1-day trend filter ensures
trades align with the daily trend to avoid counter-trend trades. Volume spikes confirm
institutional participation at reversal points. This strategy aims to capture mean
reversion moves in both bull and bear markets by trading pullbacks to the trend.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h Williams %R data - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R on 6h data
    williams_r_6h = calculate_williams_r(
        df_6h['high'].values, df_6h['low'].values, df_6h['close'].values
    )
    
    # Align Williams %R to 6h timeframe
    williams_r_6h_aligned = align_htf_to_ltf(prices, df_6h, williams_r_6h)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume average (24-period)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_6h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg_24[i])):
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
        
        williams_r = williams_r_6h_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80), above 1d EMA, volume spike
            if (williams_r < -80 and                    # Oversold
                close[i] > ema_34_1d_aligned[i] and     # Above 1d EMA (bullish trend)
                volume[i] > 2.0 * vol_avg_24[i]):       # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), below 1d EMA, volume spike
            elif (williams_r > -20 and                  # Overbought
                  close[i] < ema_34_1d_aligned[i] and   # Below 1d EMA (bearish trend)
                  volume[i] > 2.0 * vol_avg_24[i]):     # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50) or crosses 1d EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 or price crosses below 1d EMA
                if williams_r > -50 or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R falls below -50 or price crosses above 1d EMA
                if williams_r < -50 or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0