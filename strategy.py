#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Donchian breakout with volume confirmation and daily trend filter.
# Long when price breaks above 20-day high with volume > 1.5x average and daily EMA(50) rising.
# Short when price breaks below 20-day low with volume > 1.5x average and daily EMA(50) falling.
# Designed for low trade frequency (15-25/year) to minimize fee drag and capture breakouts in trending markets.

name = "12h_DonchianBreakout_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on daily
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_trend_up = ema_50_1d[1:] > ema_50_1d[:-1]  # Rising EMA
    daily_trend_up = np.concatenate([[False], daily_trend_up])  # Align with daily index
    
    # Align indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_up.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(daily_trend_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above Donchian high with volume and uptrend
            if (close[i] > donchian_high_aligned[i] and
                daily_trend_aligned[i] > 0.5 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below Donchian low with volume and downtrend
            elif (close[i] < donchian_low_aligned[i] and
                  daily_trend_aligned[i] <= 0.5 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian low or trend turns down
            if close[i] < donchian_low_aligned[i] or daily_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian high or trend turns up
            if close[i] > donchian_high_aligned[i] or daily_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals