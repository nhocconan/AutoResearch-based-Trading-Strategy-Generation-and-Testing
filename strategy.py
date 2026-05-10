#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
# Hypothesis: Breakouts at Camarilla R1/S1 levels with 12h trend filter and volume spikes.
# Long when: price breaks above R1 with 12h uptrend and volume > 2x average.
# Short when: price breaks below S1 with 12h downtrend and volume > 2x average.
# Uses 12h EMA50 for trend filter to avoid counter-trend trades.
# Volume spike confirms institutional participation.
# Works in bull/bear by following 12h trend and using volume to filter false breakouts.
# Target: 20-40 trades/year per symbol.

name = "4H_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # We'll calculate daily OHLC first, then derive Camarilla levels
    # For simplicity, we'll use rolling window to approximate daily OHLC
    # In practice, we'd resample to daily, but we avoid resampling per rules
    # Instead, we use the previous day's close, high, low from 4h data
    
    # Calculate typical price for approximation
    typical_price = (high + low + close) / 3
    
    # 24-period rolling window approximates 1 day (24*4h = 96h, too long)
    # Actually, 6 * 4h = 24h, so we use 6-period for approximate daily
    window = 6  # 6 * 4h = 24h approximate
    
    # Rolling max/min/sum for approximate OHLC
    high_max = pd.Series(high).rolling(window=window, min_periods=window).max().values
    low_min = pd.Series(low).rolling(window=window, min_periods=window).min().values
    close_prev = pd.Series(close).shift(window).values  # Previous day's close
    
    # Pivot point (approx)
    pivot = (high_max + low_min + close_prev) / 3
    
    # Camarilla levels
    range_val = high_max - low_min
    R1 = close_prev + (range_val * 1.1 / 12)
    S1 = close_prev - (range_val * 1.1 / 12)
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    bullish_12h = close_12h > ema50_12h
    bearish_12h = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, bullish_12h.astype(float))
    bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, bearish_12h.astype(float))
    
    # Volume spike detector
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=24, min_periods=24).mean().values  # 24*4h = 4d average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, window)  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isclose(high_max[i], 0) or np.isclose(low_min[i], 0) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(bullish_12h_aligned[i]) or np.isnan(bearish_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i]
        volume_spike = vol_ratio > 2.0
        
        bullish = bullish_12h_aligned[i] > 0.5
        bearish = bearish_12h_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: break above R1 with 12h uptrend and volume spike
            if bullish and volume_spike and close[i] > R1[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with 12h downtrend and volume spike
            elif bearish and volume_spike and close[i] < S1[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to pivot or trend changes
            if close[i] < pivot[i] or not bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to pivot or trend changes
            if close[i] > pivot[i] or not bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals