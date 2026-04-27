#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Band squeeze with 1-day trend filter and volume confirmation.
In bull market (price > 1-day EMA34): long when BB width < 20th percentile and price > SMA20.
In bear market (price < 1-day EMA34): short when BB width < 20th percentile and price < SMA20.
BB squeeze identifies low volatility breakout conditions, daily trend filters direction,
volume confirms institutional participation. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    daily_close = df_1d['close'].values
    ema_34_1d = np.empty_like(daily_close, dtype=np.float64)
    ema_34_1d.fill(np.nan)
    if len(daily_close) >= 34:
        alpha = 2.0 / (34 + 1)
        ema_34_1d[33] = np.mean(daily_close[:34])
        for i in range(34, len(daily_close)):
            ema_34_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Bollinger Bands (20, 2) on 4h data
    bb_period = 20
    bb_std = 2.0
    sma_20 = np.empty_like(close, dtype=np.float64)
    sma_20.fill(np.nan)
    for i in range(bb_period - 1, n):
        sma_20[i] = np.mean(close[i-bb_period+1:i+1])
    
    bb_std_dev = np.empty_like(close, dtype=np.float64)
    bb_std_dev.fill(np.nan)
    for i in range(bb_period - 1, n):
        bb_std_dev[i] = np.std(close[i-bb_period+1:i+1])
    
    bb_upper = sma_20 + bb_std * bb_std_dev
    bb_lower = sma_20 - bb_std * bb_std_dev
    bb_width = bb_upper - bb_lower
    
    # Calculate 20th percentile of BB width for squeeze detection
    bb_width_pct_20 = np.empty_like(bb_width, dtype=np.float64)
    bb_width_pct_20.fill(np.nan)
    for i in range(bb_period - 1, n):
        if i >= 50:  # Need sufficient history for percentile
            window = bb_width[i-49:i+1]
            bb_width_pct_20[i] = np.percentile(window, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need BB (20), BB width percentile (50)
    start_idx = max(bb_period - 1, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(sma_20[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bb_width_pct_20[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        sma_val = sma_20[i]
        bb_width_val = bb_width[i]
        bb_width_pct = bb_width_pct_20[i]
        ema_trend = ema_34_1d_aligned[i]
        
        # Daily close price for trend comparison
        daily_close_price = df_1d['close'].values
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close_price)
        if np.isnan(daily_close_aligned[i]):
            signals[i] = 0.0
            continue
        daily_close_val = daily_close_aligned[i]
        
        # Volume filter: volume > 1.1x average (calculated from 4h volume MA20)
        vol_ma_20 = np.empty_like(volume, dtype=np.float64)
        vol_ma_20.fill(np.nan)
        for j in range(19, n):
            vol_ma_20[j] = np.mean(volume[j-19:j+1])
        vol_filter = vol_now > 1.1 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Squeeze condition: BB width < 20th percentile of recent width
        squeeze = bb_width_val < bb_width_pct
        
        if position == 0:
            # Bull market (price > daily EMA34): look for long when squeeze + price > SMA20
            if daily_close_val > ema_trend and squeeze and price_now > sma_val and vol_filter:
                signals[i] = size
                position = 1
            # Bear market (price < daily EMA34): look for short when squeeze + price < SMA20
            elif daily_close_val < ema_trend and squeeze and price_now < sma_val and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < SMA20 or trend changes to bear
            if price_now < sma_val or daily_close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price > SMA20 or trend changes to bull
            if price_now > sma_val or daily_close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_BBSqueeze_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0