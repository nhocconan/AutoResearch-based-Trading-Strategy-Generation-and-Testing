#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h volume confirmation + ADX filter
# Donchian breakout captures breakouts in trending markets, volume confirms strength
# ADX > 25 ensures we only trade in trending conditions, avoiding whipsaws in ranges
# Works in bull markets (breakouts up) and bear markets (breakouts down)
# Designed for low trade frequency (target 20-40/year) with clear trend following
# Uses discrete position sizing to minimize fee churn

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h average volume (20-period) for confirmation
    avg_volume = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # 12h ADX(14) for trend strength
    tr1 = np.maximum(high_12h[1:], low_12h[:-1]) - np.minimum(high_12h[1:], low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 12h indicators to 4h timeframe
    avg_volume_aligned = align_htf_to_ltf(prices, df_12h, avg_volume)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_aligned[i] > 25:
            # Long breakout: price closes above 20-period high with volume confirmation
            if (close[i] > highest_high[i] and 
                volume[i] > 1.5 * avg_volume_aligned[i] and 
                position <= 0):
                position = 1
                signals[i] = position_size
            # Short breakdown: price closes below 20-period low with volume confirmation
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.5 * avg_volume_aligned[i] and 
                  position >= 0):
                position = -1
                signals[i] = -position_size
            # Exit when price returns to middle of channel
            elif position == 1 and close[i] < (highest_high[i] + lowest_low[i]) / 2:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > (highest_high[i] + lowest_low[i]) / 2:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Volume_ADX_Trend"
timeframe = "4h"
leverage = 1.0