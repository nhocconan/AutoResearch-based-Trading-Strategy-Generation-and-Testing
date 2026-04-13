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
    
    # 1-day data for pivot levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # Daily data for pivot levels (previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean()
    
    # Align all data to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w.values)
    
    # Daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 1.8x 20-period average (approximated from daily)
        # 6 4h periods per day, so divide daily MA by 6
        volume_4h_approx = volume[i]  # Current 4h bar volume
        volume_ma_20_4h = volume_ma_20_1d_aligned[i] / 6  # Approximate 20-period average for 4h
        volume_condition = volume_4h_approx > (volume_ma_20_4h * 1.8)
        
        # Trend filter: only long when price > weekly EMA20, short when price < weekly EMA20
        long_trend = close[i] > ema_20_1w_aligned[i]
        short_trend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions: price near Camarilla levels with volume and trend confirmation
        # Long when price touches or crosses above S1/S2 with volume and uptrend
        # Short when price touches or crosses below R1/R2 with volume and downtrend
        near_support = (close[i] <= s1_aligned[i] * 1.005) or (close[i] <= s2_aligned[i] * 1.005)
        near_resistance = (close[i] >= r1_aligned[i] * 0.995) or (close[i] >= r2_aligned[i] * 0.995)
        
        if position == 0:
            if near_support and volume_condition and long_trend:
                position = 1
                signals[i] = position_size
            elif near_resistance and volume_condition and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price reaches pivot or shows reversal signs
            if close[i] >= pivot_aligned[i] * 0.995:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price reaches pivot or shows reversal signs
            if close[i] <= pivot_aligned[i] * 1.005:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d1w_Camarilla_Pivot_Breakout_With_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0