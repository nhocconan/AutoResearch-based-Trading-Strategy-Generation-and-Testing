#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for pivot points and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard floor trader pivots)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x daily volume MA (adjusted for 6h)
        # 4x 6h periods per day, so daily MA/4 = approximate 6h period MA
        volume_6h_approx_ma = volume_ma_20_1d_aligned[i] / 4
        volume_condition = volume[i] > (volume_6h_approx_ma * 1.5)
        
        # Trend filter: weekly EMA20 direction
        long_trend = close[i] > ema_20_1w_aligned[i]
        short_trend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions: Fade at R3/S3, breakout continuation at R4/S4
        # R4 = R3 + (R2 - R1), S4 = S3 - (S1 - S2)
        r4 = r3_aligned[i] + (r2_aligned[i] - r1_aligned[i])
        s4 = s3_aligned[i] - (s1_aligned[i] - s2_aligned[i])
        
        # Fade at extreme levels (R3/S3) when price rejects
        fade_short = (close[i] >= r3_aligned[i]) and (close[i] < r3_aligned[i] + 0.1 * (r4 - r3_aligned[i]))
        fade_long = (close[i] <= s3_aligned[i]) and (close[i] > s3_aligned[i] - 0.1 * (s3_aligned[i] - s4))
        
        # Breakout continuation beyond R4/S4
        breakout_long = close[i] > r4
        breakout_short = close[i] < s4
        
        if position == 0:
            # Fade trades at R3/S3 with volume and counter-trend
            if fade_short and volume_condition and short_trend:
                position = -1
                signals[i] = -position_size
            elif fade_long and volume_condition and long_trend:
                position = 1
                signals[i] = position_size
            # Breakout trades beyond R4/S4 with volume and trend
            elif breakout_long and volume_condition and long_trend:
                position = 1
                signals[i] = position_size
            elif breakout_short and volume_condition and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price reaches S1 or shows rejection at resistance
            if close[i] <= s1_aligned[i] or close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short when price reaches R1 or shows rejection at support
            if close[i] >= r1_aligned[i] or close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d1w_Pivot_Fade_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0