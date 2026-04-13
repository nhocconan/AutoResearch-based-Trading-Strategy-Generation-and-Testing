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
    
    # Calculate weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    ema_20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily pivot points (standard floor trader pivots)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    r2 = pivot + (daily_high - daily_low)
    s2 = pivot - (daily_high - daily_low)
    r3 = daily_high + 2 * (pivot - daily_low)
    s3 = daily_low - 2 * (daily_high - pivot)
    r4 = daily_high + 3 * (pivot - daily_low)
    s4 = daily_low - 3 * (daily_high - pivot)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.3x daily volume MA (adjusted for 6h)
        # 4 6h periods per day, so daily MA/4 = approximate 6h period MA
        volume_6h_approx_ma = volume_ma_20_1d_aligned[i] / 4
        volume_condition = volume[i] > (volume_6h_approx_ma * 1.3)
        
        # Trend filter: weekly EMA20 direction
        long_trend = close[i] > ema_20_1w_aligned[i]
        short_trend = close[i] < ema_20_1w_aligned[i]
        
        # Price levels
        price = close[i]
        
        if position == 0:
            # Fade at R3/S3 with volume and counter-trend
            if price >= r3_aligned[i] and volume_condition and short_trend:
                position = -1
                signals[i] = -position_size
            elif price <= s3_aligned[i] and volume_condition and long_trend:
                position = 1
                signals[i] = position_size
            # Breakout continuation at R4/S4 with volume and trend
            elif price > r4_aligned[i] and volume_condition and long_trend:
                position = 1
                signals[i] = position_size
            elif price < s4_aligned[i] and volume_condition and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price returns to pivot or breaks S4
            if price <= pivot_aligned[i] or price < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price returns to pivot or breaks R4
            if price >= pivot_aligned[i] or price > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d1w_Camarilla_Pivot_Fade_Breakout_v1"
timeframe = "6h"
leverage = 1.0