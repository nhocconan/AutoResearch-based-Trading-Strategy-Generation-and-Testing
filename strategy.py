#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R + 12h EMA50 trend filter with volume confirmation.
Williams %R identifies overbought/oversold conditions. In trending markets, extreme readings
can signal continuation rather than reversal. Combined with 12h EMA50 trend filter and volume
spike confirmation, this captures momentum moves while avoiding counter-trend trades.
Designed for moderate trade frequency (~20-40/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 14-period Williams %R data (using 1h for better resolution, aligned to 4h)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1h) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1h, williams_r)
    
    # Volume confirmation: volume / 20-period average volume (1h)
    vol_ma_20 = pd.Series(df_1h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1h = df_1h['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1h, vol_ratio_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_12h_aligned[i]
        wr = williams_r_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + volume spike + uptrend
            if (wr < -80 and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) + volume spike + downtrend
            elif (wr > -20 and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R returns to neutral range (-50) or trend reversal
            if position == 1 and (wr > -50 or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (wr < -50 or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0