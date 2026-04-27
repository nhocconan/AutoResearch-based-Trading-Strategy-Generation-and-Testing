#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe strategy using weekly Donchian breakout + weekly trend filter + volume confirmation.
# Weekly Donchian breakouts capture major trend changes. Volume confirms institutional participation.
# Weekly trend filter (EMA 20) ensures alignment with higher timeframe momentum.
# This combination reduces false breakouts and works in both bull and bear markets by following the weekly trend.
# Target: 10-25 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period) from previous weekly bar
    # Donchian Upper = max(high over last 20 weeks)
    # Donchian Lower = min(low over last 20 weeks)
    donchian_upper = np.full(len(high_1w), np.nan)
    donchian_lower = np.full(len(low_1w), np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_upper[i] = np.max(high_1w[i-20:i])
        donchian_lower[i] = np.min(low_1w[i-20:i])
    
    # Shift by 1 to use only previous weekly bar's Donchian levels (no look-ahead)
    donchian_upper = np.roll(donchian_upper, 1)
    donchian_lower = np.roll(donchian_lower, 1)
    donchian_upper[0] = np.nan
    donchian_lower[0] = np.nan
    
    # Align Donchian levels to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Weekly EMA trend filter (20-period)
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: volume > 2.0 x 20-day average (daily periods)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20 weeks), EMA (20), volume MA (20)
    start_idx = max(20, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from weekly EMA
        bullish_trend = price > ema_20_aligned[i]
        bearish_trend = price < ema_20_aligned[i]
        
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper + volume + bullish weekly trend
            if price > donchian_upper and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly Donchian lower + volume + bearish weekly trend
            elif price < donchian_lower and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below weekly Donchian lower or trend turns bearish
            if price < donchian_lower or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above weekly Donchian upper or trend turns bullish
            if price > donchian_upper or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0