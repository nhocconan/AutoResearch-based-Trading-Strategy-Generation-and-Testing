#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Use weekly Camarilla R3/S3 levels as dynamic support/resistance.
Enter long when price breaks above R3 with volume spike and weekly uptrend.
Enter short when price breaks below S3 with volume spike and weekly downtrend.
Camarilla levels adapt to volatility, providing structure in both trending and ranging markets.
Volume spike confirms institutional interest. Weekly trend filter avoids counter-trend trades.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
"""
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
    
    # Get weekly data for trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on previous week's range)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2 where C=(H+L+CLOSE)/3
    # Actually standard Camarilla uses previous day's OHLC, but for weekly:
    # Use prior week's high, low, close to calculate current week's levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for pivot calculation
    typical_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla R3 and S3 levels
    r3_1w = typical_1w + range_1w * 1.1 / 2
    s3_1w = typical_1w - range_1w * 1.1 / 2
    
    # Align weekly levels to 12h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Get 12h EMA for trend confirmation (using weekly trend as filter)
    # Actually use the weekly close trend: EMA of weekly close
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get 12h volume for volume spike detection
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need weekly Camarilla, volume MA, and weekly EMA
    start_idx = max(20, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        r3 = r3_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        weekly_trend = ema_20_1w_aligned[i]
        
        # Volume filter: volume > 2.0x 12h average (strict to reduce trades)
        vol_filter = vol_now > 2.0 * vol_ma
        
        # Entry conditions: Camarilla R3/S3 breakout with volume and weekly trend alignment
        if position == 0:
            # Long: break above R3 + volume + weekly uptrend
            if close[i] > r3 and vol_filter and close[i] > weekly_trend:
                signals[i] = size
                position = 1
            # Short: break below S3 + volume + weekly downtrend
            elif close[i] < s3 and vol_filter and close[i] < weekly_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below weekly EMA or S3 level (stop and reverse possible)
            if close[i] < weekly_trend or close[i] < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above weekly EMA or R3 level
            if close[i] > weekly_trend or close[i] > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0