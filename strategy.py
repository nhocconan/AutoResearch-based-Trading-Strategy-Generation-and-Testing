#!/usr/bin/env python3
# Hypothesis: 4h Camarilla pivot S1/S3 breakout with 12h EMA trend filter and volume spike confirmation
# Long when price breaks above S3 with 12h EMA50 > EMA200 and volume > 2x average
# Short when price breaks below S1 with 12h EMA50 < EMA200 and volume > 2x average
# Exit when price crosses Camarilla pivot point (PP)
# Combines institutional pivot levels, trend alignment, and volume confirmation
# Designed for low-frequency, high-conviction trades on 4h timeframe
# Target: 75-200 total trades over 4 years (19-50/year) with size 0.25

name = "4h_Camarilla_S1S3_Breakout_12hEMA_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # EMA trend: 1 if bullish (EMA50 > EMA200), -1 if bearish
    ema_trend = np.where(ema_50 > ema_200, 1, -1)
    ema_trend_aligned = align_htf_to_ltf(prices, df_12h, ema_trend)
    
    # Calculate Camarilla levels from previous day
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: PP, S1, S3, R1, R3
    # PP = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3
    # Range = H - L
    rng = high_1d - low_1d
    # S1 = C - (Range * 1.1 / 12)
    s1 = close_1d - (rng * 1.1 / 12)
    # S3 = C - (Range * 1.1 / 4)
    s3 = close_1d - (rng * 1.1 / 4)
    # R1 = C + (Range * 1.1 / 12)
    r1 = close_1d + (rng * 1.1 / 12)
    # R3 = C + (Range * 1.1 / 4)
    r3 = close_1d + (rng * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_trend_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above S3 with bullish trend and volume spike
            if (close[i] > s3_aligned[i] and 
                ema_trend_aligned[i] > 0 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with bearish trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  ema_trend_aligned[i] < 0 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below pivot point
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above pivot point
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals