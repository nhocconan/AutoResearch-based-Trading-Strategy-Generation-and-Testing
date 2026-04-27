#!/usr/bin/env python3
"""
Hypothesis: 1-day ADX trend + Bollinger Bands mean reversion.
In trending markets (ADX > 25): buy pullbacks to upper band in uptrend, sell rallies to lower band in downtrend.
In ranging markets (ADX <= 25): fade extremes at Bollinger Bands (2,20) with weekly trend filter.
Weekly trend (price > weekly EMA50) adds confluence. Target: 15-25 trades/year per symbol.
"""

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
    
    # Get daily data for ADX and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on daily
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(daily_high[1:] - daily_low[1:])
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((daily_high[1:] - daily_high[:-1]) > (daily_low[:-1] - daily_low[1:]),
                       np.maximum(daily_high[1:] - daily_high[:-1], 0), 0)
    dm_minus = np.where((daily_low[:-1] - daily_low[1:]) > (daily_high[1:] - daily_high[:-1]),
                        np.maximum(daily_low[:-1] - daily_low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smoothed_avg(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nanmean(arr[1:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smoothed_avg(tr, 14)
    dm_plus_smooth = smoothed_avg(dm_plus, 14)
    dm_minus_smooth = smoothed_avg(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_avg(dx, 14)
    
    # Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = np.full_like(daily_close, np.nan)
    for i in range(bb_period-1, len(daily_close)):
        sma[i] = np.mean(daily_close[i-bb_period+1:i+1])
    
    bb_std_dev = np.full_like(daily_close, np.nan)
    for i in range(bb_period-1, len(daily_close)):
        bb_std_dev[i] = np.std(daily_close[i-bb_period+1:i+1])
    
    upper_band = sma + bb_std * bb_std_dev
    lower_band = sma - bb_std * bb_std_dev
    
    # Weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    ema_50_1w = np.full_like(weekly_close, np.nan)
    if len(weekly_close) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(weekly_close[:50])
        for i in range(50, len(weekly_close)):
            ema_50_1w[i] = alpha * weekly_close[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Align all indicators to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    sma_aligned = align_htf_to_ltf(prices, df_1d, sma)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_close_price = df_1w['close'].values
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close_price)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ADX (14+14=28), BB (20), weekly EMA (50)
    start_idx = max(28, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_band_aligned[i]) or
            np.isnan(lower_band_aligned[i]) or np.isnan(sma_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(weekly_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        adx_val = adx_aligned[i]
        upper_band = upper_band_aligned[i]
        lower_band = lower_band_aligned[i]
        sma_val = sma_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        weekly_close_val = weekly_close_aligned[i]
        
        # Regime filter: trending if ADX > 25, ranging if ADX <= 25
        is_trending = adx_val > 25
        is_uptrend = weekly_close_val > ema_trend  # Weekly trend
        
        if position == 0:
            if is_trending:
                # Trending: buy pullbacks to upper band in uptrend, sell rallies to lower band in downtrend
                if is_uptrend and price_now <= upper_band:
                    signals[i] = size
                    position = 1
                elif (not is_uptrend) and price_now >= lower_band:
                    signals[i] = -size
                    position = -1
            else:
                # Ranging: fade extremes at Bollinger Bands
                if price_now <= lower_band:
                    signals[i] = size
                    position = 1
                elif price_now >= upper_band:
                    signals[i] = -size
                    position = -1
            if signals[i] == 0.0:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses SMA (mean reversion) or trend change
            if price_now >= sma_val or (is_trending and not is_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses SMA or trend change
            if price_now <= sma_val or (is_trending and is_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_ADX_BB_TrendMeanRev"
timeframe = "1d"
leverage = 1.0