#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter.
# Long when price breaks above Donchian high (20) + volume > 1.5x 20-period average + 1d EMA50 up.
# Short when price breaks below Donchian low (20) + volume > 1.5x 20-period average + 1d EMA50 down.
# Exit when price crosses Donchian midline (10-period average of high/low) or opposite breakout occurs.
# Uses 4h timeframe for entries, 1d for trend filter to avoid whipsaw in sideways markets.
# Target: 20-50 trades per year to minimize fee drag while capturing strong trends.
# Works in bull markets via breakouts and in bear markets via short breakdowns.

name = "4h_Donchian20_Volume_1dTrend"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period) on 4h
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(20, len(high)):
        highest_high[i] = np.max(high[i-20:i+1])
        lowest_low[i] = np.min(low[i-20:i+1])
    
    # Donchian midline (10-period average of high/low) for exit
    highest_high_10 = np.full_like(high, np.nan)
    lowest_low_10 = np.full_like(low, np.nan)
    for i in range(10, len(high)):
        highest_high_10[i] = np.max(high[i-10:i+1])
        lowest_low_10[i] = np.min(low[i-10:i+1])
    donchian_mid = (highest_high_10 + lowest_low_10) / 2
    
    # Volume average (20-period) for confirmation
    vol_avg = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_avg[i] = np.mean(volume[i-20:i+1])
    
    # 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = ema_50_1d[1:] > ema_50_1d[:-1]
    trend_1d_up = np.concatenate([[False], trend_1d_up])
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_avg[i]) or np.isnan(trend_1d_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high + volume confirmation + 1d uptrend
            if (close[i] > highest_high[i] and 
                volume[i] > 1.5 * vol_avg[i] and 
                trend_1d_up_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < Donchian low + volume confirmation + 1d downtrend
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  not trend_1d_up_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses Donchian midline downward OR short breakdown signal
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            # Also exit if reverse breakdown occurs
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  not trend_1d_up_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses Donchian midline upward OR long breakout signal
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            # Also exit if reverse breakout occurs
            elif (close[i] > highest_high[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  trend_1d_up_aligned[i]):
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals