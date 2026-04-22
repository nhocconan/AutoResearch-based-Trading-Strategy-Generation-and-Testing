#!/usr/bin/env python3

"""
Hypothesis: 6-hour Bollinger Band breakout with 1-week trend filter and volume confirmation.
This strategy trades mean reversion in strong trends: during weekly uptrends, buy
BB lower band touches; during weekly downtrends, sell BB upper band touches.
The 1-week trend filter ensures we only trade with the higher timeframe trend.
Volume spikes confirm institutional participation.
Targets 15-35 trades/year per symbol (60-140 total over 4 years) to minimize fee drag.
Works in both bull and bear markets by aligning with the weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_bollinger_bands(close, length=20, mult=2.0):
    """Calculate Bollinger Bands: middle, upper, lower"""
    basis = pd.Series(close).rolling(window=length, min_periods=length).mean()
    dev = pd.Series(close).rolling(window=length, min_periods=length).std()
    upper = basis + mult * dev
    lower = basis - mult * dev
    return basis.values, upper.values, lower.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h Bollinger Bands data - ONCE before loop
    bb_length = 20
    bb_mult = 2.0
    basis, upper, lower = calculate_bollinger_bands(close, bb_length, bb_mult)
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 6h volume average (24-period)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg_24[i])):
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
            # Long: price touches lower BB, weekly uptrend, volume spike
            if (low[i] <= lower[i] and                    # Price touches lower BB
                close[i] > ema_34_1w_aligned[i] and       # Weekly uptrend
                volume[i] > 2.0 * vol_avg_24[i]):         # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB, weekly downtrend, volume spike
            elif (high[i] >= upper[i] and                 # Price touches upper BB
                  close[i] < ema_34_1w_aligned[i] and     # Weekly downtrend
                  volume[i] > 2.0 * vol_avg_24[i]):       # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle Bollinger Band
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above middle band
                if close[i] >= basis[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below middle band
                if close[i] <= basis[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Bollinger_Bands_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0