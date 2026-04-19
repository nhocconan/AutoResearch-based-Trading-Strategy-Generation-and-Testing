#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R overbought/oversold with weekly trend filter and volume confirmation.
# Williams %R identifies reversal points; weekly trend ensures alignment with higher timeframe momentum.
# Volume filter confirms participation. Designed for low frequency (~20-40 trades/year) to avoid fee drag.
# Long when: Williams %R < -80 (oversold), weekly close > weekly open (bullish), volume > 1.5x average.
# Short when: Williams %R > -20 (overbought), weekly close < weekly open (bearish), volume > 1.5x average.
# Exit: Opposite Williams %R level or trend reversal.

name = "6h_WilliamsR_WeeklyTrend_Volume"
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
    
    # Weekly trend: bullish if weekly close > weekly open
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_trend = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(weekly_trend[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: oversold, weekly bullish, volume confirmation
            if (williams_r[i] < -80 and 
                weekly_trend[i] > 0.5 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: overbought, weekly bearish, volume confirmation
            elif (williams_r[i] > -20 and 
                  weekly_trend[i] < 0.5 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if overbought or weekly trend turns bearish
            if (williams_r[i] > -20) or (weekly_trend[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if oversold or weekly trend turns bullish
            if (williams_r[i] < -80) or (weekly_trend[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals