#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R extremes with 1d trend filter and volume confirmation
# Long when 12h Williams %R < -80 (oversold) and price > 1d EMA50 with volume > 1.3x 20-period average
# Short when 12h Williams %R > -20 (overbought) and price < 1d EMA50 with volume > 1.3x 20-period average
# Uses Williams %R for mean reversion extremes, EMA50 for trend filter, volume for confirmation
# Designed to work in ranging markets via mean reversion and in trending markets via trend alignment
# Target: 12-30 trades per year (48-120 over 4 years) with 0.25 position sizing

name = "6h_12hWilliamsR_1dEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams %R (14-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    highest_high = df_12h['high'].rolling(window=14, min_periods=14).max().values
    lowest_low = df_12h['low'].rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_12h['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) and price above EMA50 with volume confirmation
            if williams_r_aligned[i] < -80 and close[i] > ema_50_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) and price below EMA50 with volume confirmation
            elif williams_r_aligned[i] > -20 and close[i] < ema_50_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 (mean reversion) or price below EMA50
            if williams_r_aligned[i] > -50 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 (mean reversion) or price above EMA50
            if williams_r_aligned[i] < -50 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals