#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h for signal direction (EMA50 trend) and 1h for precise entry timing.
# Camarilla R3/S3 levels provide institutional breakout/mean reversion zones.
# Volume confirmation ensures institutional participation.
# Session filter (08-20 UTC) reduces noise from low-liquidity periods.
# Designed for 15-37 trades/year on 1h to minimize fee drag while capturing trends.
# Works in bull markets via breakouts and in bear markets via mean reversion at extremes.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for Camarilla levels (more stable than 4h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's data
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_range = (high_1d - low_1d)
    camarilla_r3 = close_1d + (1.1 * camarilla_range / 2)
    camarilla_s3 = close_1d - (1.1 * camarilla_range / 2)
    
    # Align Camarilla levels to 1h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume spike filter (20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 with volume spike and 4h EMA50 uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 with volume spike and 4h EMA50 downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla range (mean reversion) OR trend reverses
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price re-enters Camarilla range OR trend reverses
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals