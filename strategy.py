#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above 1h Camarilla R3 level AND 4h EMA50 > 4h EMA200 AND volume > 1.5x 20-period average
# Short when price breaks below 1h Camarilla S3 level AND 4h EMA50 < 4h EMA200 AND volume > 1.5x 20-period average
# Exit when price crosses 1h Camarilla pivot point (mean reversion) OR 4h EMA50/200 crossover reverses
# Uses 1h primary timeframe with 4h HTF for EMA trend filter (more responsive than 1d for intraday swings)
# Camarilla levels provide clear breakout zones based on previous hour's range
# EMA filter ensures we only trade in trending markets on 4h, reducing whipsaw in ranges
# Volume confirmation filters low-momentum breakouts
# Discrete sizing (0.20) to limit fee drag and manage drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Session filter: 08-20 UTC to avoid low-volume Asian session noise

name = "1h_Camarilla_R3S3_Breakout_4hEMA_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 and EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align EMAs to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Get 4h data ONCE before loop for Camarilla levels (based on previous 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla R3 and S3 levels for 4h: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3_4h = close_4h + (1.1 * (high_4h - low_4h) / 2)
    camarilla_s3_4h = close_4h - (1.1 * (high_4h - low_4h) / 2)
    camarilla_pivot_4h = (high_4h + low_4h + close_4h) / 3  # Standard pivot point
    
    # Align to 1h timeframe (using previous 4h bar's levels)
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    camarilla_pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(camarilla_r3_4h_aligned[i]) or 
            np.isnan(camarilla_s3_4h_aligned[i]) or 
            np.isnan(camarilla_pivot_4h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 4h EMA50 > EMA200 AND volume spike
            if (close[i] > camarilla_r3_4h_aligned[i] and 
                ema50_4h_aligned[i] > ema200_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 4h EMA50 < EMA200 AND volume spike
            elif (close[i] < camarilla_s3_4h_aligned[i] and 
                  ema50_4h_aligned[i] < ema200_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion) OR 4h EMA50 < EMA200 (trend weakening)
            if close[i] < camarilla_pivot_4h_aligned[i] or ema50_4h_aligned[i] < ema200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion) OR 4h EMA50 > EMA200 (trend weakening)
            if close[i] > camarilla_pivot_4h_aligned[i] or ema50_4h_aligned[i] > ema200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals