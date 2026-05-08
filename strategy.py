#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets; EMA50 filter ensures alignment with higher-timeframe trend.
# Volume confirmation filters false breakouts. Works in bull markets (riding uptrends) and bear markets (riding downtrends).
# Target: 100-200 total trades over 4 years (25-50/year) with disciplined entries.

name = "4h_Donchian_20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels (20-period)
    upper = np.full(len(high_4h), np.nan)
    lower = np.full(len(low_4h), np.nan)
    for i in range(20, len(high_4h)):
        upper[i] = np.max(high_4h[i-20:i])
        lower[i] = np.min(low_4h[i-20:i])
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA50 for trend filter
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_20[i] = np.mean(volume_12h[i-20:i])
    
    # Align all indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or \
           np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 12h bar's volume
            idx_12h = 0
            while idx_12h < len(df_12h) and df_12h.iloc[idx_12h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_12h += 1
            idx_12h -= 1  # last completed 12h bar
            
            if idx_12h >= 0:
                vol_12h_current = df_12h.iloc[idx_12h]['volume']
                vol_filter = vol_12h_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout + trend + volume confirmation
            # Long when price breaks above upper band in uptrend (price > EMA50) with volume
            long_condition = (close[i] > upper_aligned[i]) and \
                             (close[i] > ema50_aligned[i]) and vol_filter
            # Short when price breaks below lower band in downtrend (price < EMA50) with volume
            short_condition = (close[i] < lower_aligned[i]) and \
                              (close[i] < ema50_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower band or trend fails
            if (close[i] < lower_aligned[i]) or (close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper band or trend fails
            if (close[i] > upper_aligned[i]) or (close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals