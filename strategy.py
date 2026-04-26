#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime
Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend filter (price > EMA50) and volume regime (volume > 1.5x 50-period median). 
Enters long on close above R1 when 1d trend bullish and volume elevated. 
Enters short on close below S1 when 1d trend bearish and volume elevated.
Uses discrete position sizing (0.25) to minimize churn. Designed for 50-150 total trades over 4 years.
Works in both bull and bear markets by following 1d trend filter.
"""

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
    
    # Calculate Camarilla levels for 12h timeframe using prior bar's OHLC
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # Use prior bar to avoid look-ahead
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    prior_high[0] = high[0]  # first bar uses current
    prior_low[0] = low[0]
    prior_close[0] = close[0]
    
    camarilla_range = prior_high - prior_low
    r1 = prior_close + camarilla_range * 1.1 / 12
    s1 = prior_close - camarilla_range * 1.1 / 12
    
    # Volume regime: volume > 1.5x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_regime = volume > (1.5 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period volume median and prior bar)
    start_idx = max(50, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_median[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: close above R1 + volume regime + bullish 1d trend
        if close[i] > r1[i] and volume_regime[i] and close[i] > ema_50_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: close below S1 + volume regime + bearish 1d trend
        elif close[i] < s1[i] and volume_regime[i] and close[i] < ema_50_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite Camarilla level touch
        elif position == 1 and close[i] < s1[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime"
timeframe = "12h"
leverage = 1.0