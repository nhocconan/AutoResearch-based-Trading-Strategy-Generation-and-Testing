#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and ADX trend filter
# Uses 20-period Donchian channels on 12h timeframe for breakout signals.
# Volume confirmation ensures breakouts are supported by participation.
# ADX > 25 filters for trending markets, avoiding false breakouts in ranging conditions.
# Works in bull markets (upward breakouts) and bear markets (downward breakouts).
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Timeframe: 12h, HTF: 1w for trend context

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for higher timeframe trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period) on 12h for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(close, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(close, 1)), 
                        np.maximum(np.roll(close, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx[i]) or np.isnan(ema_50_1w_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian upper band + volume confirmation + 
        # ADX > 25 + price above 1w EMA50 (bullish higher timeframe bias)
        if (close[i] > highest_high[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            adx[i] > 25 and
            close[i] > ema_50_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian lower band + volume confirmation + 
        # ADX > 25 + price below 1w EMA50 (bearish higher timeframe bias)
        elif (close[i] < lowest_low[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              adx[i] > 25 and
              close[i] < ema_50_1w_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < lowest_low[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_high[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Volume_ADX_1wEMA_Filter"
timeframe = "12h"
leverage = 1.0