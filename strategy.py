#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Works in both bull/bear markets by
# requiring alignment with daily trend and high-volume to confirm reversals.
# Target: 50-150 trades over 4 years (12-37/year) with position size 0.25.
name = "6h_WilliamsR_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA20 trend filter
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_6h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate Williams %R on 6h timeframe (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_6h[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80  # Oversold condition
        overbought = williams_r[i] > -20  # Overbought condition
        
        trend_up = close[i] > ema_20_6h[i]
        trend_down = close[i] < ema_20_6h[i]
        
        if position == 0:
            # Long: oversold + uptrend + volume confirmation
            if oversold and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought + downtrend + volume confirmation
            elif overbought and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: overbought or trend reversal
            if overbought or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: oversold or trend reversal
            if oversold or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals