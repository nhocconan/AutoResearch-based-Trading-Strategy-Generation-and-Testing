#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Uses 1d Camarilla pivot levels (R3/S3) for breakout entries on 12h timeframe.
# Trend filter: 1d EMA34. Volume confirmation: volume > 20-period average.
# Targets 12-30 trades/year to avoid fee drag while capturing strong moves.
# Works in bull/bear via breakout logic and trend alignment.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # Using typical pivot: (H + L + C) / 3
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # Actually, standard Camarilla:
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    # We'll use R3 and S3 as breakout levels
    
    # Calculate for each day, using previous day's HLC
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    range_1d = high_1d[:-1] - low_1d[:-1]
    r3 = pivot + (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 4)
    
    # Shift to align with current day (today's levels based on yesterday)
    # For current day i, we use levels from day i-1
    r3_full = np.concatenate([[np.nan], r3])  # prepend NaN for first day
    s3_full = np.concatenate([[np.nan], s3])
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_full)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_full)
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volume confirmation
            vol_ok = volume[i] > vol_ma[i]
            
            # Long: break above R3 + price above EMA34 + volume
            if (close[i] > r3_12h[i] and 
                close[i] > ema_34_12h[i] and vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 + price below EMA34 + volume
            elif (close[i] < s3_12h[i] and 
                  close[i] < ema_34_12h[i] and vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below R3 or trend changes
            if (close[i] < r3_12h[i] or 
                close[i] < ema_34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above S3 or trend changes
            if (close[i] > s3_12h[i] or 
                close[i] > ema_34_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals