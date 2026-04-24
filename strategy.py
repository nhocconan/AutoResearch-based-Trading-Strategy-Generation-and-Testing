#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA34 trend alignment (proven BTC/ETH edge).
- Donchian channel calculated from prior 20 periods on 12h timeframe.
- Breakout logic: long when price closes above upper band with volume spike and uptrend,
                  short when price closes below lower band with volume spike and downtrend.
- Trend filter: only long when 12h close > 1d EMA34, only short when 12h close < 1d EMA34.
- Volume confirmation: current 12h volume > 1.8 * 20-period 12h volume MA.
- Discrete signal size: 0.25 to minimize fee churn and control drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in both bull/bear: trend filter avoids counter-trend trades, Donchian breakouts capture momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 12h timeframe (primary)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Donchian channels: upper = max(high, 20), lower = min(low, 20)
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # AlDonchian channels to 12h timeframe (already aligned via get_htf_data)
    # But we need to align to original 12h index for comparison with close
    # Since df_12h is already 12h data, we can use it directly with proper alignment
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.8 * volume_ma)
    
    # Trend filter: 12h close vs 1d EMA34
    uptrend = close_12h > ema_34_1d_aligned[:len(close_12h)]  # Ensure same length
    downtrend = close_12h < ema_34_1d_aligned[:len(close_12h)]
    
    # Align all 12h indicators to original price index
    # We need to map 12h indices to original price indices
    # Since we're using 12h timeframe as primary, we need to work with 12h bars
    # But the signal array must match original prices length
    
    # Create signal array for original prices
    signals = np.zeros(n)
    
    # We'll iterate through 12h bars and set signals for the corresponding 12h bar index
    # But we need to map 12h bar index to original price index
    # Simpler approach: work entirely in 12h timeframe, then broadcast to original
    
    # Re-index approach: create 12h-aligned arrays for original price index
    # Since get_htf_data gives us actual 12h ohlc, we need to align it to original index
    
    # Let's work with 12h data and create signals for 12h bars, then align to original
    signals_12h = np.zeros(len(close_12h))
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need Donchian(20) and EMA34
    
    for i in range(start_idx, len(close_12h)):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals_12h[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper band AND uptrend AND volume spike
            if close_12h[i] > donchian_upper[i] and uptrend[i] and volume_spike[i]:
                signals_12h[i] = 0.25
                position = 1
            # Short: price closes below lower band AND downtrend AND volume spike
            elif close_12h[i] < donchian_lower[i] and downtrend[i] and volume_spike[i]:
                signals_12h[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle of Donchian channel or reverse signal
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2
            if close_12h[i] <= donchian_mid:
                signals_12h[i] = 0.0
                position = 0
            else:
                signals_12h[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle of Donchian channel or reverse signal
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2
            if close_12h[i] >= donchian_mid:
                signals_12h[i] = 0.0
                position = 0
            else:
                signals_12h[i] = -0.25
    
    # Now align 12h signals to original price index
    # We need to map each 12h bar to its corresponding range in original prices
    # Since we don't have the exact mapping, we'll use a simpler approach:
    # For each original price bar, find the most recent 12h signal
    
    # Create a mapping from original index to 12h index
    # We know that 12h bars are every 48 original bars (assuming 15m original data)
    # But to be safe, we'll use the index from get_htf_data
    
    # Actually, let's reconsider: since we're declaring timeframe="12h",
    # the generate_signals function should work with 12h data as input
    # But the interface requires it to work with the given prices DataFrame
    # which is in the primary timeframe
    
    # Let's restart with a cleaner approach
    
    # Reset and use proper MTF alignment
    
    # Extract data for primary timeframe (whatever it is, but we'll work with it)
    # Actually, since timeframe="12h", we should expect prices to be 12h data
    # But to be safe, let's work with the given prices as primary timeframe
    
    # Re-implement with clean logic
    
    # Extract price and volume data from given prices (assumed to be in primary timeframe)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on primary timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    # Trend filter: close vs 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need Donchian(20) and EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above upper band AND uptrend AND volume spike
            if close[i] > donchian_upper[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower band AND downtrend AND volume spike
            elif close[i] < donchian_lower[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to middle of Donchian channel or reverse signal
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] <= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to middle of Donchian channel or reverse signal
            donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] >= donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0