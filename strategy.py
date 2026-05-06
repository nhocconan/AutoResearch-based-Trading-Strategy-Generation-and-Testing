#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# Uses Donchian breakout for entry, 1d EMA50 for trend filter, volume >1.5x20 for confirmation
# Exit on opposite Donchian(20) touch or trailing stop using 3x ATR
# Designed for 4h timeframe targeting 75-200 trades over 4 years (19-50/year)
# Works in both bull/bear: breaks capture trends, volume filter avoids false breakouts

name = "4h_Donchian20_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for 4h timeframe (for trailing stop)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(volume_filter[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_long = 0.0
                lowest_since_short = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND above 1d EMA50 AND volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_long = close[i]
            # Short entry: price breaks below Donchian low AND below 1d EMA50 AND volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_short = close[i]
        elif position == 1:
            # Update highest since entry
            highest_since_long = max(highest_since_long, close[i])
            # Exit conditions:
            # 1. Price touches Donchian low (opposite channel)
            # 2. Trailing stop: price drops 3*ATR from highest since entry
            if close[i] < donchian_low[i] or close[i] < highest_since_long - 3.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
                highest_since_long = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest since entry
            lowest_since_short = min(lowest_since_short, close[i])
            # Exit conditions:
            # 1. Price touches Donchian high (opposite channel)
            # 2. Trailing stop: price rises 3*ATR from lowest since entry
            if close[i] > donchian_high[i] or close[i] > lowest_since_short + 3.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
                lowest_since_short = 0.0
            else:
                signals[i] = -0.25
    
    return signals