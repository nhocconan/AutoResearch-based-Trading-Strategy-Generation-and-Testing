#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Uses Donchian channels for breakout direction, 1d EMA50 for trend filter, and volume spike for confirmation.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets by taking breakouts in the direction of the 1d trend.
# Target: 20-40 trades/year to avoid fee drag and ensure robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on daily close with proper handling
    ema_50_1d = np.full(len(df_1d), np.nan)
    if len(close_1d) >= 50:
        # Use pandas EMA for accuracy and proper NaN handling
        ema_series = pd.Series(close_1d).ewm(span=50, adjust=False).mean()
        ema_50_1d = ema_series.values
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 2.0 * 20-period average (higher threshold to reduce trades)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for Donchian (20) and EMA50
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA50
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_50_1d_aligned[i-1]):
            trend_up = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            trend_down = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + uptrend + volume spike
            if (close[i] > highest_high[i] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + downtrend + volume spike
            elif (close[i] < lowest_low[i] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian lower or trend turns down
            if (close[i] < lowest_low[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper or trend turns up
            if (close[i] > highest_high[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0