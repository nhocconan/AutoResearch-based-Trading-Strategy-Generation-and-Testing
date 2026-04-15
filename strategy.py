#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with daily EMA filter and volume spike
# Williams %R (14) identifies overbought/oversold conditions.
# In trending markets (price > daily EMA50), we buy oversold dips (%R < -80).
# In ranging markets (price near daily EMA50), we sell overbought bounces (%R > -20).
# Volume spike confirms participation. Designed to work in both bull and bear markets
# by adapting to the trend via daily EMA filter. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume spike: current volume > 2.0 * median of last 20 bars
    volume_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    volume_spike = volume > (2.0 * volume_median)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i])):
            continue
        
        # Determine market regime based on price vs daily EMA50
        price_vs_ema = close[i] - ema_50_1d_aligned[i]
        ema_threshold = 0.01 * ema_50_1d_aligned[i]  # 1% of EMA level
        
        # Trending market: significant deviation from EMA
        if abs(price_vs_ema) > ema_threshold:
            # In uptrend: buy oversold dips
            if price_vs_ema > 0 and williams_r[i] < -80 and volume_spike[i] and position <= 0:
                position = 1
                signals[i] = base_size
            # In downtrend: sell overbought bounces
            elif price_vs_ema < 0 and williams_r[i] > -20 and volume_spike[i] and position >= 0:
                position = -1
                signals[i] = -base_size
        # Ranging market: price near EMA
        else:
            # Sell overbought bounces
            if williams_r[i] > -20 and volume_spike[i] and position >= 0:
                position = -1
                signals[i] = -base_size
            # Buy oversold dips
            elif williams_r[i] < -80 and volume_spike[i] and position <= 0:
                position = 1
                signals[i] = base_size
        
        # Exit: opposite Williams %R level or loss of volume spike
        if position == 1 and (williams_r[i] > -20 or not volume_spike[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (williams_r[i] < -80 or not volume_spike[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_EMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0