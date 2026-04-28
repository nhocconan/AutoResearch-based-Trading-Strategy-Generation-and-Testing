#!/usr/bin/env python3
"""
12h_ThreeBar_Low_High_Breakout_1dTrend_Volume
Hypothesis: On 12-hour timeframe, enter long when price breaks above the 3-period high (HH3) with volume surge and 1d uptrend (price above 1d EMA50), short when price breaks below the 3-period low (LL3) with volume surge and 1d downtrend. Exit on opposite break (LL3 for longs, HH3 for shorts). Uses 1d EMA50 trend filter to avoid counter-trend trades. Designed for low trade frequency (~15-30/year) to minimize fee decay in both bull and bear markets. HH3/LL3 provides dynamic support/resistance that adapts to volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Trend: bullish when price > EMA50, bearish when price < EMA50
    d1_uptrend = close > ema_50_aligned
    d1_downtrend = close < ema_50_aligned
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 1.5)
    
    # 3-period high (HH3) and low (LL3) - dynamic support/resistance
    hh3 = pd.Series(high).rolling(window=3, min_periods=3).max().values
    ll3 = pd.Series(low).rolling(window=3, min_periods=3).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(hh3[i]) or np.isnan(ll3[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with 1d EMA50 trend alignment and volume surge
        long_entry = close[i] > hh3[i] and d1_uptrend[i] and volume_surge[i]
        short_entry = close[i] < ll3[i] and d1_downtrend[i] and volume_surge[i]
        
        # Exit on opposite break (LL3 for longs, HH3 for shorts)
        long_exit = close[i] < ll3[i] and volume_surge[i]
        short_exit = close[i] > hh3[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_ThreeBar_Low_High_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0