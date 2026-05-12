#!/usr/bin/env python3
# 1d_DonchianBreakout_WeeklyTrend
# Hypothesis: Use weekly Donchian breakout with 1d trend filter and volume confirmation.
# Enter long when price breaks above weekly Donchian high and close > 1d EMA50,
# enter short when price breaks below weekly Donchian low and close < 1d EMA50.
# Weekly trend filter prevents counter-trend trades. Volume confirmation ensures breakout strength.
# Designed for low frequency (10-30 trades/year) to avoid fee drag. Works in bull (catch breakouts)
# and bear (catch breakdowns) with trend filter.

name = "1d_DonchianBreakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, period):
    """
    Calculate Donchian channels.
    Returns upper band, lower band, and middle band.
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(n):
        if i >= period - 1:
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20 periods)
    donchian_high, donchian_low = donchian_channels(high_1w, low_1w, 20)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly data to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        trend_up = close[i] > ema_50_1d[i]
        trend_down = close[i] < ema_50_1d[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        if position == 0:
            # LONG: breakout above weekly Donchian high, price above daily EMA50, volume confirmation
            if breakout_up and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: breakout below weekly Donchian low, price below daily EMA50, volume confirmation
            elif breakout_down and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price breaks below weekly Donchian low or trend fails
            if close[i] < donchian_low_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above weekly Donchian high or trend fails
            if close[i] > donchian_high_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals