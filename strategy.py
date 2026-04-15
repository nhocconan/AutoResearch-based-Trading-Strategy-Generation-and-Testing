#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and ADX regime filter
# Donchian(20) breakouts capture momentum; volume confirms institutional interest;
# ADX > 20 ensures trending market (avoids chop). Works in bull/bear via direction.
# 12h timeframe targets 12-37 trades/year to minimize fee drag.
# Signal size 0.25 balances return and drawdown (e.g., 77% BTC drop → ~19% equity loss).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for HTF context (trend, channels)
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily
    donchian_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for completed daily bar)
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 14-period ADX on daily for regime filter
    # True Range
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - daily_close_prev)
    tr3 = np.abs(daily_low - daily_close_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])) > 
                       (np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low), 
                       np.maximum(daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low) > 
                        (daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])), 
                        np.maximum(np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low, 0), 0)
    
    # Smoothed DM
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = daily_volume / (vol_ma + 1e-10)
    
    # Align volume ratio to 12h timeframe
    vol_ratio_12h = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(vol_ratio_12h[i])):
            signals[i] = 0.0
            continue
        
        # Long: price breaks above Donchian high in uptrend (ADX > 20) with volume confirmation
        if (close[i] > donchian_high_12h[i] and 
            adx_12h[i] > 20 and 
            vol_ratio_12h[i] > 1.5):  # Volume 50% above average
            signals[i] = 0.25
        # Short: price breaks below Donchian low in downtrend (ADX > 20) with volume confirmation
        elif (close[i] < donchian_low_12h[i] and 
              adx_12h[i] > 20 and 
              vol_ratio_12h[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0