#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Uses 12h EMA50 for trend direction to capture intermediate-term momentum
# Uses 4h volume > 2.0x 20-period EMA for strong confirmation to filter weak breakouts
# Designed for 4h timeframe targeting 20-50 trades/year with discrete sizing (0.25)
# Works in bull markets (breakouts with volume in uptrend) and bear markets (breakouts with volume in downtrend)
# Camarilla R3/S3 levels provide institutional pivot points with statistical significance
# 12h timeframe balances responsiveness with noise reduction vs 1d

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot calculation (more stable than lower timeframes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, S3, R4, S4
    # R3 = Close + 1.1*(High - Low)/2
    # S3 = Close - 1.1*(High - Low)/2
    # R4 = Close + 1.1*(High - Low)
    # S4 = Close - 1.1*(High - Low)
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 2
    s3 = close_1d - 1.1 * camarilla_range / 2
    r4 = close_1d + 1.1 * camarilla_range
    s4 = close_1d - 1.1 * camarilla_range
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 4h data for volume EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h volume EMA(20) for volume confirmation
    vol_4h = df_4h['volume'].values
    vol_ema_20 = pd.Series(vol_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: price breaks above R3 + volume confirmation + 12h EMA > EMA50 (uptrend)
            if (close[i] > r3_aligned[i] and volume_confirmed and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume confirmation + 12h EMA < EMA50 (downtrend)
            elif (close[i] < s3_aligned[i] and volume_confirmed and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 (mean reversion) OR 12h EMA < EMA50 (trend change)
            if close[i] < s3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 (mean reversion) OR 12h EMA > EMA50 (trend change)
            if close[i] > r3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals