#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly trend filter (1w EMA200) for bias,
# 6h Williams %R mean reversion for entry, and volume confirmation.
# Uses 1d price action for stop/reversal conditions.
# Designed to work in both bull and bear by following weekly trend while fading extremes.
# Targets 15-25 trades/year (60-100 total over 4 years) with strict entry conditions.
name = "6h_1w_EMA200_WilliamsR_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (00-24 UTC for 6h - all hours)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = np.ones(n, dtype=bool)  # 6h: trade all hours
    
    # Get 1w data for EMA200 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get 6h data for Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend bias
        weekly_bullish = close[i] > ema_200_1w_aligned[i]
        weekly_bearish = close[i] < ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: weekly bullish AND Williams %R oversold (< -80) with volume
            if (weekly_bullish and 
                williams_r[i] < -80 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly bearish AND Williams %R overbought (> -20) with volume
            elif (weekly_bearish and 
                  williams_r[i] > -20 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if weekly turns bearish OR Williams %R overbought (> -20)
            if not weekly_bullish or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if weekly turns bullish OR Williams %R oversold (< -80)
            if not weekly_bearish or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals