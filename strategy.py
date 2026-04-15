#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Intraday Range Breakout with Volume Spike and ADX Trend Filter
# Uses the session high/low (UTC 00:00-12:00 and 12:00-00:00) as dynamic support/resistance.
# Breakouts are traded only with volume > 2x average and ADX > 25 (trending market).
# Works in bull markets (breakouts up) and bear markets (breakouts down).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Determine session: 0-11 = first 12h (00:00-12:00 UTC), 12-23 = second 12h (12:00-00:00 UTC)
    hours = prices.index.hour
    session = (hours // 12).astype(int)  # 0 for first 12h, 1 for second 12h
    
    # Calculate session high/low
    session_high = np.full(n, np.nan)
    session_low = np.full(n, np.nan)
    
    for i in range(n):
        if i == 0 or session[i] != session[i-1]:
            # New session
            session_high[i] = high[i]
            session_low[i] = low[i]
        else:
            # Continue session
            session_high[i] = max(session_high[i-1], high[i])
            session_low[i] = min(session_low[i-1], low[i])
    
    # Load 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
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
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(session_high[i]) or np.isnan(session_low[i]) or
            np.isnan(adx_aligned[i])):
            continue
        
        # Volume condition: current volume > 2x median of last 20 bars
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_ok = volume[i] > 2.0 * vol_median
        
        # Long entry: price breaks above session high + volume + ADX > 25
        if (close[i] > session_high[i] and
            volume_ok and
            adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below session low + volume + ADX > 25
        elif (close[i] < session_low[i] and
              volume_ok and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < session_low[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > session_high[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Intraday_Range_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0