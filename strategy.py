#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R(14) with 1w trend filter and volume confirmation
# Long when Williams %R crosses above -20 from below with 1w bullish trend (close > EMA50) and volume > 1.8x 20-period volume EMA
# Short when Williams %R crosses below -80 from above with 1w bearish trend (close < EMA50) and volume > 1.8x 20-period volume EMA
# Williams %R identifies overbought/oversold conditions with reversal signals
# Volume spike filter (1.8x) is strict to minimize trades and reduce fee drag
# Uses 1w EMA50 for major trend alignment to avoid counter-trend whipsaw
# Target: 15-30 trades/year on 1d timeframe to stay within fee-efficient range

name = "1d_WilliamsR_1wTrend_VolumeConfirm"
timeframe = "1d"
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
    
    # Align 1w trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Calculate Williams %R (14-period) from 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Calculate Williams %R previous value for crossover detection
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan  # First value has no previous
    
    # Bullish crossover: Williams %R crosses above -20 from below
    bullish_crossover = (williams_r > -20) & (williams_r_prev <= -20)
    # Bearish crossover: Williams %R crosses below -80 from above
    bearish_crossover = (williams_r < -80) & (williams_r_prev >= -80)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.8)  # Volume at least 1.8x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(bullish_crossover[i]) or np.isnan(bearish_crossover[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R bullish crossover AND 1w bullish trend AND volume spike
            if (bullish_crossover[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1w bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R bearish crossover AND 1w bearish trend AND volume spike
            elif (bearish_crossover[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1w bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R bearish crossover OR 1w trend turns bearish
            if (bearish_crossover[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R bullish crossover OR 1w trend turns bullish
            if (bullish_crossover[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals