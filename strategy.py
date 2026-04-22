#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R with 1-day trend filter and volume confirmation.
Long when Williams %R < -80 (oversold), 1-day EMA50 rising, and volume > 1.5x average.
Short when Williams %R > -20 (overbought), 1-day EMA50 falling, and volume > 1.5x average.
Exit when Williams %R crosses back through -50 or trend reverses.
Williams %R identifies momentum extremes; 1-day EMA50 filters higher timeframe trend.
Designed for low trade frequency by requiring oversold/overbought conditions + volume + trend alignment.
Works in both bull and bear markets by fading extremes in the direction of the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R (14 periods)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for Williams %R and volume MA
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), 1-day EMA50 rising, and volume filter
            if (williams_r[i] < -80 and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), 1-day EMA50 falling, and volume filter
            elif (williams_r[i] > -20 and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 OR trend reverses
                if (williams_r[i] > -50 or 
                    ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 OR trend reverses
                if (williams_r[i] < -50 or 
                    ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0