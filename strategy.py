#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Fibonacci Retracement + Volume Spike + Daily Trend
# Hypothesis: Buy pullbacks to 0.618 Fib in uptrend, sell rallies to 0.382 Fib in downtrend,
# confirmed by volume spike. Works in bull/bear by trading with daily trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_fib_retracement_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and swing points
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Daily close for trend filter
    close_daily = df_daily['close'].values
    
    # Daily EMA(50) for trend filter (more stable than 20)
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Calculate daily swing high/low for Fibonacci retracement
    # Use 20-day lookback for swing points
    lookback = 20
    highest_high = pd.Series(close_daily).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(close_daily).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate Fibonacci levels
    diff = highest_high - lowest_low
    fib_0236 = lowest_low + diff * 0.236  # Shallow retracement
    fib_0382 = lowest_low + diff * 0.382  # Medium retracement
    fib_0500 = lowest_low + diff * 0.500  # Medium retracement
    fib_0618 = lowest_low + diff * 0.618  # Deep retracement (golden zone)
    fib_0786 = lowest_low + diff * 0.786  # Deep retracement
    
    # Align Fibonacci levels to 6h
    fib_0236_6h = align_htf_to_ltf(prices, df_daily, fib_0236)
    fib_0382_6h = align_htf_to_ltf(prices, df_daily, fib_0382)
    fib_0500_6h = align_htf_to_ltf(prices, df_daily, fib_0500)
    fib_0618_6h = align_htf_to_ltf(prices, df_daily, fib_0618)
    fib_0786_6h = align_htf_to_ltf(prices, df_daily, fib_0786)
    
    # Volume filter: 6x average volume (strong spike)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_6h[i]) or np.isnan(fib_0382_6h[i]) or np.isnan(fib_0618_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: at least 2x average volume
        vol_ok = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 1:  # Long position
            # Exit: price reaches 0.236 Fib (take profit) or trend changes
            if high[i] >= fib_0236_6h[i] or close[i] < ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches 0.786 Fib (take profit) or trend changes
            if low[i] <= fib_0786_6h[i] or close[i] > ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Look for retracement entries in direction of daily trend
            if vol_ok:
                if close[i] > ema_50_6h[i]:  # Uptrend - buy retracements
                    # Enter near 0.618 Fib (golden zone) with rejection
                    if low[i] <= fib_0618_6h[i] and close[i] > fib_0618_6h[i]:
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend - sell retracements
                    # Enter near 0.382 Fib with rejection
                    if high[i] >= fib_0382_6h[i] and close[i] < fib_0382_6h[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals