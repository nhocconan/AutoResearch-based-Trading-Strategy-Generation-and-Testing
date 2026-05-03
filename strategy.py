#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band breakout with 1d trend filter and volume confirmation
# Bollinger Bands provide dynamic support/resistance that adapts to volatility.
# 1d EMA50 trend filter ensures trades align with higher timeframe direction.
# Volume confirmation (1.5x 20-period EMA) filters low-conviction breakouts.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.
# Works in both bull and bear markets by only taking breakouts in the direction of 1d trend.

name = "12h_BB20_2_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands (20, 2) from previous 12h bar
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = (sma_20 + 2 * std_20).shift(1).values
    bb_lower = (sma_20 - 2 * std_20).shift(1).values
    
    # Volume confirmation: 20-period EMA on 12h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid BB and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Uptrend: price above 1d EMA50
        uptrend = close[i] > ema_50_aligned[i]
        # Downtrend: price below 1d EMA50
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper BB in uptrend with volume spike
            if close[i] > bb_upper[i] and uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB in downtrend with volume spike
            elif close[i] < bb_lower[i] and downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower BB or loses uptrend
            if close[i] < bb_lower[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper BB or loses downtrend
            if close[i] > bb_upper[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals