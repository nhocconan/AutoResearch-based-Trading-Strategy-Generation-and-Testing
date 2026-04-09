#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1w EMA trend filter + volume confirmation
# Williams %R(14) identifies overbought/oversold conditions. In strong uptrends (price > 1w EMA50),
# we buy pullbacks to %R < -80. In strong downtrends (price < 1w EMA50), we sell rallies to %R > -20.
# Volume confirmation ensures breakouts have conviction. Designed for 6h timeframe to capture
# multi-day swings in both bull and bear markets with controlled trade frequency.

name = "6h_1w_williamsr_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close) / (highest_high - lowest_low),
                          -50)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: uptrend if price > 1w EMA50, downtrend if price < 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if Williams %R exits oversold territory or trend changes
            if williams_r[i] > -50 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if Williams %R exits overbought territory or trend changes
            if williams_r[i] < -50 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long in uptrend on Williams %R oversold with volume confirmation
            if uptrend and williams_r[i] < -80 and volume_confirmed[i]:
                position = 1
                signals[i] = 0.25
            # Enter short in downtrend on Williams %R overbought with volume confirmation
            elif downtrend and williams_r[i] > -20 and volume_confirmed[i]:
                position = -1
                signals[i] = -0.25
    
    return signals