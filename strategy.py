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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels (R1, S1)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align daily levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: volume > 1.8x 20-period average (stricter than before)
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # ADX filter for trend strength (14-period)
    adx = np.full(n, np.nan)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = np.full(n, np.nan)
    for i in range(13, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    plus_di_14 = np.full(n, np.nan)
    minus_di_14 = np.full(n, np.nan)
    for i in range(13, n):
        plus_di_14[i] = 100 * np.mean(plus_dm[i-13:i+1]) / atr_14[i] if atr_14[i] != 0 else 0
        minus_di_14[i] = 100 * np.mean(minus_dm[i-13:i+1]) / atr_14[i] if atr_14[i] != 0 else 0
    
    dx = np.full(n, np.nan)
    for i in range(13, n):
        if (plus_di_14[i] + minus_di_14[i]) != 0:
            dx[i] = 100 * abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])
    
    adx = np.full(n, np.nan)
    for i in range(26, n):
        adx[i] = np.mean(dx[i-12:i+1])
    
    adx_aligned = adx  # ADX is already on 12h timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h EMA (50 periods), daily data, volume MA (20 periods), ADX (26 periods)
    start_idx = max(50, 20, 26)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_trend = ema_50_12h_aligned[i]
        pivot_level = pivot_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: volume > 1.8x average (stricter)
        vol_filter = vol_now > 1.8 * vol_avg
        
        # ADX filter: only trade when trend is strong (ADX > 25)
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long: price breaks above R1 + bullish trend + volume spike + strong trend
            if price > r1_level and price > ema_trend and vol_filter and trend_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 + bearish trend + volume spike + strong trend
            elif price < s1_level and price < ema_trend and vol_filter and trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot (mean reversion) or trend turns bearish or ADX weakens
            if price <= pivot_level or price < ema_trend or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot (mean reversion) or trend turns bullish or ADX weakens
            if price >= pivot_level or price > ema_trend or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume_ADX"
timeframe = "12h"
leverage = 1.0