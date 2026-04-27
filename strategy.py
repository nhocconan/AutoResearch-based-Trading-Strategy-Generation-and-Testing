#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter and weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Weekly pivot levels (from weekly candle)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot and S1/R1 levels
    pivot_w = (high_1w + low_1w + close_1w) / 3.0
    r1_w = close_1w + (high_1w - low_1w) * 1.1 / 12
    s1_w = close_1w - (high_1w - low_1w) * 1.1 / 12
    
    # Align weekly levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA (200), weekly pivot, daily volume MA (20)
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(pivot_w_aligned[i]) or 
            np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(vol_avg_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_trend = ema_200_1w_aligned[i]
        pivot_w = pivot_w_aligned[i]
        r1_w = r1_w_aligned[i]
        s1_w = s1_w_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_avg_20d_aligned[i]
        
        # Volume filter: volume > 1.5x 20-day average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price above weekly R1 + above weekly EMA200 + volume spike
            if price > r1_w and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price below weekly S1 + below weekly EMA200 + volume spike
            elif price < s1_w and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly pivot or trend turns bearish
            if price <= pivot_w or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to weekly pivot or trend turns bullish
            if price >= pivot_w or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_Pivot_R1S1_Breakout_EMA200_Trend_Volume"
timeframe = "6h"
leverage = 1.0