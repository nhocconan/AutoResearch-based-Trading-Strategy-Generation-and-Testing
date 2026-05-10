#!/usr/bin/env python3
# 4h_PriceAction_SR_12hTrend_Volume
# Hypothesis: Price action at key support/resistance levels from 12h swing points
# combined with 12h EMA trend filter and volume confirmation creates high-probability
# entries with low frequency. Works in bull markets via bounces off support in uptrend
# and in bear markets via rejections at resistance in downtrend.
# Target: 20-40 trades/year on 4h timeframe.

name = "4h_PriceAction_SR_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_swing_points(high, low, window=3):
    """Calculate swing highs and lows: swing high is highest high in window, 
    swing low is lowest low in window"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Swing high: current high is the highest in the last 'window' periods
    swing_high = high_series.rolling(window=2*window+1, center=True).max()
    # Swing low: current low is the lowest in the last 'window' periods
    swing_low = low_series.rolling(window=2*window+1, center=True).min()
    
    # Only keep points where current high/low equals the rolling max/min
    swing_high = np.where(high_series.values == swing_high.values, swing_high.values, np.nan)
    swing_low = np.where(low_series.values == swing_low.values, swing_low.values, np.nan)
    
    return swing_high, swing_low

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for swing points and trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate swing points on 12h timeframe
    swing_high_12h, swing_low_12h = calculate_swing_points(
        df_12h['high'].values, 
        df_12h['low'].values, 
        window=2
    )
    
    # Calculate 12h EMA for trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(
        span=21, adjust=False, min_periods=21
    ).mean().values
    
    # Align 12h data to 4h timeframe
    swing_high_12h_aligned = align_htf_to_ltf(prices, df_12h, swing_high_12h)
    swing_low_12h_aligned = align_htf_to_ltf(prices, df_12h, swing_low_12h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 4h data for price action and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(
        span=20, adjust=False, min_periods=20
    ).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need swing points calculation
    start_idx = 5  # enough for swing point calculation
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(swing_high_12h_aligned[i]) or 
            np.isnan(swing_low_12h_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        if position == 0:
            # Long setup: price at 12h swing low support in uptrend with volume
            if (not np.isnan(swing_low_12h_aligned[i]) and 
                low[i] <= swing_low_12h_aligned[i] * 1.005 and  # within 0.5% of support
                uptrend and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: price at 12h swing high resistance in downtrend with volume
            elif (not np.isnan(swing_high_12h_aligned[i]) and 
                  high[i] >= swing_high_12h_aligned[i] * 0.995 and  # within 0.5% of resistance
                  downtrend and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches 12h swing high resistance or trend changes
            if (not np.isnan(swing_high_12h_aligned[i]) and 
                high[i] >= swing_high_12h_aligned[i] * 0.995) or \
               not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches 12h swing low support or trend changes
            if (not np.isnan(swing_low_12h_aligned[i]) and 
                low[i] <= swing_low_12h_aligned[i] * 1.005) or \
               not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals