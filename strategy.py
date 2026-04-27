#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for high/low range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range and midpoint
    daily_range = high_1d - low_1d
    daily_mid = (high_1d + low_1d) / 2.0
    
    # Align to 4h
    daily_range_aligned = align_htf_to_ltf(prices, df_1d, daily_range)
    daily_mid_aligned = align_htf_to_ltf(prices, df_1d, daily_mid)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 50-period EMA on weekly
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: volume > 1.5x 24-period average (6 hours worth of 4h bars)
    vol_ma_24 = np.full(n, np.nan, dtype=np.float64)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data and volume MA (24 periods)
    start_idx = max(24, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(daily_range_aligned[i]) or np.isnan(daily_mid_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        daily_range_val = daily_range_aligned[i]
        daily_mid_val = daily_mid_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Calculate bands: midpoint ± 0.4 * daily range
        upper_band = daily_mid_val + 0.4 * daily_range_val
        lower_band = daily_mid_val - 0.4 * daily_range_val
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper band + bullish weekly trend + volume spike
            if price > upper_band and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band + bearish weekly trend + volume spike
            elif price < lower_band and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint (mean reversion) or trend turns bearish
            if price <= daily_mid_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midpoint (mean reversion) or trend turns bullish
            if price >= daily_mid_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DailyMidBand_Breakout_1wEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0