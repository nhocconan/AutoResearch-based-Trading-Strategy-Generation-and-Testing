#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R for overbought/oversold conditions and 1w EMA for trend filter.
# Williams %R identifies reversal points in ranging markets while EMA filter ensures trades align with higher timeframe trend.
# Volume confirmation filters out low-conviction moves.
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by using 1w EMA trend filter to avoid counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Load 1w data ONCE for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA on 1w data
    close_1w = df_1w['close'].values
    ema_period = 20
    ema_1w = pd.Series(close_1w).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Align indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Williams %R and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 1w EMA
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            # Look for Williams %R reversals
            # Only trade in direction of higher timeframe trend
            
            # Long: Williams %R oversold (< -80) AND price above 1w EMA (uptrend)
            if (williams_r_aligned[i] < -80 and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) AND price below 1w EMA (downtrend)
            elif (williams_r_aligned[i] > -20 and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to overbought (> -20) or trend changes
            if (williams_r_aligned[i] > -20 or 
                below_ema):  # Price below 1w EMA (trend change)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to oversold (< -80) or trend changes
            if (williams_r_aligned[i] < -80 or 
                above_ema):  # Price above 1w EMA (trend change)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dWilliamsR_1wEMA_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0