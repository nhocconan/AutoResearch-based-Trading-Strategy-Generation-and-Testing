#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12-hour Donchian breakout with 1-day volume confirmation and 1-day trend filter.
# Long when price breaks above 12h Donchian high with volume > 1.5x daily average and price above daily EMA50.
# Short when price breaks below 12h Donchian low with volume > 1.5x daily average and price below daily EMA50.
# Designed for low trade frequency (10-20/year) to minimize whipsaw and capture strong momentum moves.

name = "6h_Donchian12h_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    donch_high = np.full_like(high_12h, np.nan)
    donch_low = np.full_like(low_12h, np.nan)
    
    for i in range(20, len(high_12h)):
        donch_high[i] = np.max(high_12h[i-20:i])
        donch_low[i] = np.min(low_12h[i-20:i])
    
    # Align Donchian levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Get daily data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily average volume (20-period SMA)
    vol_sma_20 = np.full_like(volume_1d, np.nan)
    for i in range(20, len(volume_1d)):
        vol_sma_20[i] = np.mean(volume_1d[i-20:i])
    
    # Align daily filters to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_sma_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_sma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above 12h Donchian high with volume confirmation and uptrend
            if (close[i] > donch_high_aligned[i] and
                volume[i] > vol_sma_aligned[i] * 1.5 and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below 12h Donchian low with volume confirmation and downtrend
            elif (close[i] < donch_low_aligned[i] and
                  volume[i] > vol_sma_aligned[i] * 1.5 and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below 12h Donchian low or trend turns down
            if close[i] < donch_low_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above 12h Donchian high or trend turns up
            if close[i] > donch_high_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals