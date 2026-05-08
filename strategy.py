#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12-hour Donchian breakout with volume confirmation and 1-day trend filter.
# The strategy captures medium-term breakouts aligned with higher timeframe momentum.
# Long when price breaks above 12h Donchian high with volume spike and 1-day uptrend.
# Short when price breaks below 12h Donchian low with volume spike and 1-day downtrend.
# Uses 1-day EMA(34) for trend filter to ensure alignment with daily momentum.
# Designed for low trade frequency (15-30/year) to minimize fee drag and capture high-probability breakouts.

name = "4h_DonchianBreakout_12hVolume_1dTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channel (20-period) on 12h data
    donchian_high = np.zeros_like(high_12h)
    donchian_low = np.zeros_like(low_12h)
    
    for i in range(len(high_12h)):
        if i < 19:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high_12h[i-19:i+1])
            donchian_low[i] = np.min(low_12h[i-19:i+1])
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising daily EMA
    daily_trend_up = np.concatenate([[False], daily_trend_up])  # Align with daily index
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_up.astype(float))
    
    # Volume confirmation: current volume > 2.0x 20-period EMA on 4h
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(daily_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above 12h Donchian high in uptrend with volume
            if (daily_trend_aligned[i] > 0.5 and  # Daily uptrend
                close[i] > donchian_high_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below 12h Donchian low in downtrend with volume
            elif (daily_trend_aligned[i] <= 0.5 and  # Daily downtrend
                  close[i] < donchian_low_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below 12h Donchian low or trend turns down
            if close[i] < donchian_low_aligned[i] or daily_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above 12h Donchian high or trend turns up
            if close[i] > donchian_high_aligned[i] or daily_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals