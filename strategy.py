#!/usr/bin/env python3

"""
Hypothesis: 4-hour Bollinger Band squeeze breakout with 1-day trend filter and volume confirmation.
The Bollinger Band squeeze identifies low volatility periods that precede explosive moves.
Breakouts from the squeeze with volume confirmation and 1-day trend alignment capture
strong momentum moves in both bull and bear markets. The squeeze acts as a volatility
filter, reducing false breakouts and improving win rate.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for Bollinger Bands - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on 4h data (20-period, 2 std dev)
    close_4h = df_4h['close'].values
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Align Bollinger Bands to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb)
    bb_width_aligned = align_htf_to_ltf(prices, df_4h, bb_width)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter (50-period)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Bollinger Band squeeze threshold (20th percentile of width)
    # Use expanding window to avoid look-ahead
    bb_width_series = pd.Series(bb_width_aligned)
    bb_width_percentile = bb_width_series.expanding(min_periods=50).quantile(0.20).values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_avg_20[i]) or np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bollinger Band squeeze condition: low volatility environment
        is_squeeze = bb_width_aligned[i] <= bb_width_percentile[i]
        
        if position == 0:
            # Long: breakout above upper BB, in squeeze, above 1d EMA, volume spike
            if (close[i] > upper_bb_aligned[i] and      # Breakout above upper BB
                is_squeeze and                          # In volatility squeeze
                close[i] > ema_50_1d_aligned[i] and     # Above 1d EMA (bullish trend)
                volume[i] > 2.0 * vol_avg_20[i]):       # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower BB, in squeeze, below 1d EMA, volume spike
            elif (close[i] < lower_bb_aligned[i] and    # Breakout below lower BB
                  is_squeeze and                        # In volatility squeeze
                  close[i] < ema_50_1d_aligned[i] and   # Below 1d EMA (bearish trend)
                  volume[i] > 2.0 * vol_avg_20[i]):     # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle of Bollinger Bands or crosses 1d EMA
            exit_signal = False
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            
            if position == 1:
                # Exit long: price returns to middle BB or below 1d EMA
                if close[i] < middle_bb or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle BB or above 1d EMA
                if close[i] > middle_bb or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Bollinger_Squeeze_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0