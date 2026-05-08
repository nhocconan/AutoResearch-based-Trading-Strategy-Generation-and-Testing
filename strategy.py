#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Donchian breakouts with volume confirmation and 1-week trend filter.
# Long when price breaks above prior day's Donchian high with volume and weekly uptrend.
# Short when price breaks below prior day's Donchian low with volume and weekly downtrend.
# Exit when price reverts to prior day's Donchian median or trend reverses.
# Designed for 20-30 trades/year to minimize fee drag and capture multi-day trends.

name = "4h_DonchianBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's Donchian channels (20-period)
    donch_high = np.zeros_like(close_1d)
    donch_low = np.zeros_like(close_1d)
    donch_mid = np.zeros_like(close_1d)
    
    for i in range(20, len(close_1d)):
        donch_high[i] = np.max(high_1d[i-20:i])
        donch_low[i] = np.min(low_1d[i-20:i])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # First 20 days have insufficient data
    donch_high[:20] = np.nan
    donch_low[:20] = np.nan
    donch_mid[:20] = np.nan
    
    # Align daily Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend_up = ema_34_1w[1:] > ema_34_1w[:-1]  # Rising weekly EMA
    weekly_trend_up = np.concatenate([[False], weekly_trend_up])  # Align with daily index
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above prior day's Donchian high with volume and weekly uptrend
            if (weekly_trend_aligned[i] > 0.5 and  # Weekly uptrend
                close[i] > donch_high_aligned[i] and  # Break above Donchian high
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below prior day's Donchian low with volume and weekly downtrend
            elif (weekly_trend_aligned[i] <= 0.5 and  # Weekly downtrend
                  close[i] < donch_low_aligned[i] and  # Break below Donchian low
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to Donchian mid or weekly trend turns down
            if close[i] < donch_mid_aligned[i] or weekly_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to Donchian mid or weekly trend turns up
            if close[i] > donch_mid_aligned[i] or weekly_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals