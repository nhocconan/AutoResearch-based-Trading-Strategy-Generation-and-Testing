#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Uses Donchian(20) channels on 4h for breakout detection
# Requires breakout above/below channel with close confirmation
# Uses 1d EMA(50) to filter for trend direction only
# Volume confirmation (>1.5x 20-bar average) ensures participation
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year)
# Works in both bull/bear: follows trends, avoids false signals in consolidation

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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels on 4h (20-period)
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA(50) trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) for 4h timeframe (for stoploss)
    tr1_4h = np.abs(high_4h[1:] - low_4h[1:])
    tr2_4h = np.abs(high_4h[1:] - close_4h[:-1])
    tr3_4h = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe (primary)
    highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper Donchian channel AND above 1d EMA50 AND volume confirmation
            if (close[i] > highest_20_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian channel AND below 1d EMA50 AND volume confirmation
            elif (close[i] < lowest_20_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Update trailing stop: highest high since entry
            # Exit long: price closes below 2x ATR trailing stop from highest high
            if i > 0:
                # Track highest high since entry (simplified: use rolling max of high)
                pass  # Simplified exit logic
            # Exit long: price closes below lower Donchian channel
            if close[i] < lowest_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper Donchian channel
            if close[i] > highest_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals