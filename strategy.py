#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot levels with volume confirmation and 1d EMA34 trend filter
# - Long when price breaks above S3 level with volume expansion and price above 1d EMA34
# - Short when price breaks below R3 level with volume expansion and price below 1d EMA34
# - Exit when price crosses back below/above 1d EMA34
# - Volume filter requires current volume > 1.5x 20-period average
# - Session filter: only trade between 08:00-20:00 UTC to avoid low-volume periods
# - Designed to capture institutional breakouts while avoiding whipsaws in ranging markets
# - Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing
# - Uses Camarilla levels (S3/R3) as significant support/resistance levels that often trigger strong moves when broken

name = "1h_Camarilla_S3R3_Breakout_1dEMA34_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (S3/R3) from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    # Camarilla levels: S3 = close - (high - low) * 1.1/2, R3 = close + (high - low) * 1.1/2
    s3 = close_4h - (range_4h * 1.1 / 2)
    r3 = close_4h + (range_4h * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe (previous 4h bar's levels)
    s3_1h = align_htf_to_ltf(prices, df_4h, s3)
    r3_1h = align_htf_to_ltf(prices, df_4h, r3)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 1h timeframe
    ema_34_1d_1h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filters (1h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(s3_1h[i]) or np.isnan(r3_1h[i]) or 
            np.isnan(ema_34_1d_1h[i]) or np.isnan(volume_filter[i])):
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
            # Long breakout: price breaks above S3 with volume expansion and above EMA34
            if close[i] > s3_1h[i] and volume_filter[i] and close[i] > ema_34_1d_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short breakout: price breaks below R3 with volume expansion and below EMA34
            elif close[i] < r3_1h[i] and volume_filter[i] and close[i] < ema_34_1d_1h[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema_34_1d_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema_34_1d_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals