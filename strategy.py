#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1w trend filter and volume spike
# Long when Williams %R < -80 (oversold) AND 1w bullish trend (close > EMA50) AND volume > 1.5x 20-period volume EMA
# Short when Williams %R > -20 (overbought) AND 1w bearish trend (close < EMA50) AND volume > 1.5x 20-period volume EMA
# Williams %R identifies exhaustion points in both bull and bear markets. The 1w EMA50 filter ensures we only take
# reversals in the direction of the weekly trend, reducing whipsaw. Volume confirmation adds confirmation.
# Target: 12-30 trades/year on 6h timeframe with discrete sizing (0.25) to minimize fee drag.

name = "6h_WilliamsR_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1w = close_1w > ema_50_1w
    trend_bearish_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 6h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R thresholds: < -80 oversold, > -20 overbought
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(williams_oversold[i]) or np.isnan(williams_overbought[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold AND 1w bullish trend AND volume spike
            if (williams_oversold[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1w bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought AND 1w bearish trend AND volume spike
            elif (williams_overbought[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1w bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 (momentum shift) OR 1w trend turns bearish
            if (williams_r[i] > -50 or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 (momentum shift) OR 1w trend turns bullish
            if (williams_r[i] < -50 or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals