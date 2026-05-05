#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 12h trend filter and volume spike confirmation
# Long when: Williams %R(14) < -80 (oversold) AND 12h EMA50 trend is up (close > EMA50) AND volume > 1.8x 20-period MA
# Short when: Williams %R(14) > -20 (overbought) AND 12h EMA50 trend is down (close < EMA50) AND volume > 1.8x 20-period MA
# Exit when: Williams %R returns to -50 level OR opposite extreme occurs
# Uses mean reversion in extremes with trend filter to avoid counter-trend trades, volume for conviction
# Timeframe: 6h, HTF: 12h. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WilliamsR_12hEMA50_Trend_VolumeConfirm"
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
    
    # Calculate Williams %R on 6h
    if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when high == low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 12h data ONCE before loop for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close
    close_12h = df_12h['close'].values
    if len(close_12h) >= 50:
        ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
        # Trend: up when close > EMA50, down when close < EMA50
        trend_up = close_12h > ema_50_12h
        trend_down = close_12h < ema_50_12h
    else:
        ema_50_12h = np.full(len(df_12h), np.nan)
        trend_up = np.full(len(df_12h), False)
        trend_down = np.full(len(df_12h), False)
    
    # Align 12h EMA50 trend to 6h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) + 12h uptrend + volume filter
            if (williams_r[i] < -80 and 
                trend_up_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (> -20) + 12h downtrend + volume filter
            elif (williams_r[i] > -20 and 
                  trend_down_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR short extreme occurs
            if (williams_r[i] >= -50 or williams_r[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR long extreme occurs
            if (williams_r[i] <= -50 or williams_r[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals