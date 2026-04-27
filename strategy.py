#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 144-bar Donchian breakout with 1-week EMA200 trend filter and volume confirmation.
# Donchian channel identifies breakout of 12-period high/low (144 hours = 6 days).
# Long when price breaks above upper Donchian band with 1-week EMA200 uptrend and volume.
# Short when price breaks below lower Donchian band with 1-week EMA200 downtrend and volume.
# Uses 1-week EMA200 for trend filter to align with higher timeframe direction.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drift.
# Works in bull markets (captures sustained uptrends) and bear markets (captures sustained downtrends).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1-week EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 144-bar Donchian channel (12h * 12 = 144 bars = 6 days)
    donchian_period = 144
    upper_band = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = donchian_period + 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price above upper Donchian band, 1w EMA200 uptrend, volume
        if (close[i] > upper_band[i] and 
            ema200_1w_aligned[i] > ema200_1w_aligned[i-1] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price below lower Donchian band, 1w EMA200 downtrend, volume
        elif (close[i] < lower_band[i] and 
              ema200_1w_aligned[i] < ema200_1w_aligned[i-1] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal on 1w EMA200
        elif position == 1 and ema200_1w_aligned[i] <= ema200_1w_aligned[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and ema200_1w_aligned[i] >= ema200_1w_aligned[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian144_1wEMA200_Trend_Volume"
timeframe = "12h"
leverage = 1.0