#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# Uses weekly EMA21 for trend direction (long when close > EMA21, short when close < EMA21)
# and daily Donchian channels (20-period) for breakout entries.
# Volume > 1.5x 20-period average confirms breakout strength.
# Trend filter avoids counter-trend trades, reducing whipsaw in bear markets.
# Target: 15-25 trades/year to minimize fee decay while capturing major trends.
# Focus on BTC/ETH as primary assets with proven trend-following edge.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA21 for trend filter
    close_weekly = df_weekly['close'].values
    ema_21_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_21_weekly)
    
    # Get daily data for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Upper band: 20-period high
    upper = np.full(len(high_daily), np.nan)
    lower = np.full(len(low_daily), np.nan)
    for i in range(20, len(high_daily)):
        upper[i] = np.max(high_daily[i-20:i])
        lower[i] = np.min(low_daily[i-20:i])
    
    upper_aligned = align_htf_to_ltf(prices, df_daily, upper)
    lower_aligned = align_htf_to_ltf(prices, df_daily, lower)
    
    # 20-period average volume for confirmation (daily)
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(vol_period, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_weekly_aligned[i]) or
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from weekly EMA21
        uptrend = price > ema_21_weekly_aligned[i]
        downtrend = price < ema_21_weekly_aligned[i]
        
        # Volume confirmation: > 1.5x average
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian in uptrend
            if uptrend and price > upper_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below lower Donchian in downtrend
            elif downtrend and price < lower_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below lower Donchian or trend reverses
            if price < lower_aligned[i] or price < ema_21_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above upper Donchian or trend reverses
            if price > upper_aligned[i] or price > ema_21_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian_20_WeeklyEMA21_Trend_Volume"
timeframe = "1d"
leverage = 1.0