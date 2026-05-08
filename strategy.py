# State your hypothesis:
# Strategy combines 4h Donchian breakout with daily volatility filter and volume confirmation
# Donchian channels provide clear breakout levels that work in both trending and ranging markets
# Daily ATR filter ensures we only trade during periods of elevated volatility
# Volume confirmation ensures institutional participation in breakouts
# We target 20-40 trades/year to minimize fee drag while maintaining statistical significance
# Entry conditions are strict to avoid overtrading, with clear exit rules based on opposite band touch

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_DonchianBreakout_DailyVol_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR calculation with Wilder's smoothing
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[0:14])  # Initial SMA
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR ratio: current ATR / 20-period average ATR
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / atr_ma
    atr_ratio = np.where(atr_ma == 0, 0, atr_ratio)  # Avoid division by zero
    
    # Get daily ATR ratio aligned to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channel (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(lookback, 20)  # Ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio_val = atr_ratio_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + volatility filter + volume confirmation
            if (high[i] > highest_high[i] and 
                atr_ratio_val > 1.2 and vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + volatility filter + volume confirmation
            elif (low[i] < lowest_low[i] and 
                  atr_ratio_val > 1.2 and vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches or breaks below Donchian low
            if low[i] <= lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches or breaks above Donchian high
            if high[i] >= highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals