#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter (EMA21) and volume confirmation.
# Long when price breaks above R3 in 4h uptrend (price > EMA21).
# Short when price breaks below S3 in 4h downtrend (price < EMA21).
# Volume must be > 1.5x 20-period MA to confirm breakout strength.
# Uses discrete sizing 0.20 to minimize fee churn. Target: 60-150 total trades over 4 years.
# Session filter: 08-20 UTC to reduce noise trades.
# Strategy uses 4h/1d for signal direction, 1h only for entry timing.

name = "1h_Camarilla_R3S3_4hEMA21_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h EMA21
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Get 1d data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels from 1d OHLC (using previous day's close)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.roll(close_1d, 1)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    range_1d = high_1d_shifted - low_1d_shifted
    r3 = close_1d_shifted + range_1d * 1.1 / 4
    s3 = close_1d_shifted - range_1d * 1.1 / 4
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_21_4h_aligned[i]   # 4h uptrend
        trend_down = close_val < ema_21_4h_aligned[i]  # 4h downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price breaks above R3 AND 4h uptrend AND volume spike
            if close_val > r3_aligned[i] and trend_up and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND 4h downtrend AND volume spike
            elif close_val < s3_aligned[i] and trend_down and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR 4h trend turns down
            if close_val < s3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R3 OR 4h trend turns up
            if close_val > r3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals