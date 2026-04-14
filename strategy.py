#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA trend filter + volume confirmation
# Williams %R(14) < -80 for oversold, > -20 for overbought
# 1d EMA(50) determines trend: price > EMA = uptrend, price < EMA = downtrend
# Volume filter: current volume > 1.5 * 20-period average volume
# Only take longs in uptrend when oversold, shorts in downtrend when overbought
# Williams %R exits at -50 level (mean reversion midpoint)
# Designed for 12h timeframe to reduce trade frequency and avoid fee drag
# Works in both bull and bear markets: trend filter prevents counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(50)
    ema_len = 50
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=ema_len, adjust=False, min_periods=ema_len).values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams %R on 12h data
    williams_len = 14
    highest_high = pd.Series(high).rolling(window=williams_len, min_periods=williams_len).max().values
    lowest_low = pd.Series(low).rolling(window=williams_len, min_periods=williams_len).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume filter: volume > 1.5 * 20-period average
    vol_ma_len = 20
    vol_ma = pd.Series(volume).rolling(window=vol_ma_len, min_periods=vol_ma_len).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, williams_len, vol_ma_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price vs 1d EMA
        uptrend = price > ema_1d_aligned[i]
        downtrend = price < ema_1d_aligned[i]
        
        # Williams %R signals
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_level = williams_r[i] > -50  # For long exit
        exit_level_short = williams_r[i] < -50  # For short exit
        
        if position == 0:
            # Enter long: uptrend + oversold + volume confirmation
            if uptrend and oversold and volume_filter[i]:
                position = 1
                signals[i] = position_size
            # Enter short: downtrend + overbought + volume confirmation
            elif downtrend and overbought and volume_filter[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above EMA OR Williams %R exits oversold
            if price < ema_1d_aligned[i] or williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below EMA OR Williams %R exits overbought
            if price > ema_1d_aligned[i] or williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dEMA_WilliamsR_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0