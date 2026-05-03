#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h EMA50 for trend direction and 1h Camarilla R3/S3 levels for precise entries.
# Volume spike confirms breakout conviction. Designed for 15-30 trades/year on 1h to minimize fee drag.
# Works in both bull and bear markets by trading breakouts in the direction of the 4h trend.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla levels (using previous day's OHLC)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous bar for pivot calc
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using previous day's OHLC
        # Need to get previous day's high, low, close
        prev_day_idx = i - 1
        while prev_day_idx >= 0 and pd.Timestamp(open_time[prev_day_idx]).date() == pd.Timestamp(open_time[i]).date():
            prev_day_idx -= 1
        
        if prev_day_idx < 0:
            # Not enough data for previous day, skip
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get previous day's OHLC
        phigh = np.max(high[max(0, prev_day_idx):i])
        plow = np.min(low[max(0, prev_day_idx):i])
        pclose = close[i-1]  # Previous bar's close as proxy for previous day's close
        
        # Calculate Camarilla levels
        rang = phigh - plow
        if rang <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        camarilla_r3 = pclose + rang * 1.1 / 4
        camarilla_s3 = pclose - rang * 1.1 / 4
        
        # Volume confirmation: 20-period EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Breakout conditions
        if position == 0:
            # Long: price breaks above Camarilla R3 in 4h uptrend with volume spike
            if close[i] > camarilla_r3 and ema_50_4h_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 in 4h downtrend with volume spike
            elif close[i] < camarilla_s3 and ema_50_4h_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns below Camarilla R3 or loses 4h uptrend
            if close[i] < camarilla_r3 or ema_50_4h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns above Camarilla S3 or loses 4h downtrend
            if close[i] > camarilla_s3 or ema_50_4h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals