#!/usr/bin/env python3
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
    
    # Load weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for long-term trend
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Load daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R3/S3 and R4/S4)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r3 = close_1d + range_ * 1.1 / 4   # Resistance level 3
    s3 = close_1d - range_ * 1.1 / 4   # Support level 3
    r4 = close_1d + range_ * 1.1 / 2   # Resistance level 4
    s4 = close_1d - range_ * 1.1 / 2   # Support level 4
    
    # Align all levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 24-period average (4 days worth for 6h)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_avg_24[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R4 with volume AND above weekly EMA200 (uptrend)
            if (close[i] > r4_aligned[i] and volume[i] > 2.0 * vol_avg_24[i] and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 with volume AND below weekly EMA200 (downtrend)
            elif (close[i] < s4_aligned[i] and volume[i] > 2.0 * vol_avg_24[i] and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite S3/R3 level (mean reversion)
            if position == 1:
                if not np.isnan(s3_aligned[i]) and close[i] < s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not np.isnan(r3_aligned[i]) and close[i] > r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Camarilla_R3_S4_Exit_WeeklyEMA200_Trend_Volume_Session"
timeframe = "6h"
leverage = 1.0