#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data for 6-hour candle aggregation (using 4h data as proxy)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 6-hour high/low from 4h data (each 6h candle = 1.5 4h candles)
    high_6h = np.maximum(high_4h, np.roll(high_4h, 1))  # Simple approximation
    low_6h = np.minimum(low_4h, np.roll(low_4h, 1))
    
    # Daily pivot points from 4h data (using previous day's range)
    # For 6h timeframe, we use the previous 4h period's high/low/close
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align all daily-derived indicators to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_4h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_4h, s2)
    
    # Volume confirmation using 4h volume
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(high_6h[i]) or 
            np.isnan(low_6h[i]) or np.isnan(close_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = vol_4h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and bullish weekly trend
            if (price > r1_aligned[i] and 
                vol > 1.5 * vol_ma_4h_aligned[i] and 
                price > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation and bearish weekly trend
            elif (price < s1_aligned[i] and 
                  vol > 1.5 * vol_ma_4h_aligned[i] and 
                  price < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below pivot or volume drops significantly
            if price < pivot_aligned[i] or vol < 0.5 * vol_ma_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above pivot or volume drops significantly
            if price > pivot_aligned[i] or vol < 0.5 * vol_ma_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1S1_Breakout_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0