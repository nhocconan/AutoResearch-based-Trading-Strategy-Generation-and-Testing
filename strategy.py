#!/usr/bin/env python3
# 6h_VolumeBreakout_With_1dTrend_and_Consolidation
# Hypothesis: Strong breakouts from consolidation zones with volume confirmation continue in the direction of the higher timeframe trend.
# In both bull and bear markets, major moves often begin with a volatility contraction (low volume/range) followed by expansion.
# We use 1-day EMA50 for trend direction, 6-hour Bollinger Bands to detect consolidation (band width < 50th percentile),
# and enter on breakouts with volume > 2x 24-period average. This avoids false breakouts and captures sustained moves.
# Works in bull markets (breaks above resistance in uptrend) and bear markets (breaks below support in downtrend).

name = "6h_VolumeBreakout_With_1dTrend_and_Consolidation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2) on 6h chart for consolidation detection
    close_s = pd.Series(close)
    bb_middle = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Percentile lookback for consolidation (use 50 periods lookback, require 50 for percentile)
    def rolling_percentile(arr, window, percentile):
        """Calculate rolling percentile - simplified version using pandas"""
        s = pd.Series(arr)
        return s.rolling(window=window, min_periods=window).quantile(percentile/100.0).values
    
    bb_width_percentile = rolling_percentile(bb_width, 50, 50)  # 50th percentile (median)
    
    # Volume confirmation: 24-period average (4 days on 6h chart)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for BB (20), volume MA (24), percentile (50), trend (50)
    start_idx = max(20, 24, 50, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bb_width[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1-day EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Consolidation filter: Bollinger Band width below median (low volatility)
        consolidation = bb_width[i] < bb_width_percentile[i]
        
        # Volume confirmation: significant volume spike
        volume_spike = volume[i] > volume_ma[i] * 2.0
        
        # Breakout conditions
        breakout_up = close[i] > bb_upper[i]
        breakout_down = close[i] < bb_lower[i]
        
        if position == 0:
            # Long breakout: uptrend + consolidation + volume spike + break above upper BB
            if uptrend and consolidation and volume_spike and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short breakout: downtrend + consolidation + volume spike + break below lower BB
            elif downtrend and consolidation and volume_spike and breakout_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: breakdown below middle band or trend reversal
            if close[i] < bb_middle[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: breakout above middle band or trend reversal
            if close[i] > bb_middle[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals