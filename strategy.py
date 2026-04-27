#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50) and daily structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d data for daily range calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily range for position sizing (volatility normalized)
    daily_range = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        daily_range[i] = high_1d[i] - low_1d[i]
    
    # 20-period average daily range
    avg_daily_range = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        avg_daily_range[i] = np.mean(daily_range[i-19:i+1])
    
    avg_daily_range_aligned = align_htf_to_ltf(prices, df_1d, avg_daily_range)
    
    # Get 6h data for price structure (Support/Resistance from daily range)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 10:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Volume filter: volume > 1.5x 20-period average (6h)
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d EMA (50), 1d avg range (20), volume MA (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_daily_range_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_trend_1d = ema_50_1d_aligned[i]
        avg_range = avg_daily_range_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter: price relative to daily EMA50
        bullish_trend = price > ema_trend_1d
        bearish_trend = price < ema_trend_1d
        
        # Dynamic support/resistance based on daily range
        # Support: EMA50 - 0.5 * avg_daily_range
        # Resistance: EMA50 + 0.5 * avg_daily_range
        support_level = ema_trend_1d - 0.5 * avg_range
        resistance_level = ema_trend_1d + 0.5 * avg_range
        
        if position == 0:
            # Long: price breaks above resistance + bullish trend + volume spike
            if price > resistance_level and bullish_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below support + bearish trend + volume spike
            elif price < support_level and bearish_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to EMA50 (mean reversion) or trend turns bearish
            if price <= ema_trend_1d or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to EMA50 (mean reversion) or trend turns bullish
            if price >= ema_trend_1d or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Dynamic_SR_EMA50_Volume_Filter"
timeframe = "6h"
leverage = 1.0