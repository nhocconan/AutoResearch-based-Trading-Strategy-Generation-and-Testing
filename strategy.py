#!/usr/bin/env python3
# 1h_4d_1d_camarilla_volume_regime_v1
# Strategy: 1h Camarilla pivot with volume confirmation and regime filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Camarilla levels act as strong intraday support/resistance. 
# In ranging markets (high chop), price reverts from H3/L3 levels. 
# In trending markets (low chop), price breaks H4/L4 levels.
# Volume confirms institutional interest. Uses 4h trend + 1d regime filter.
# Designed for 15-30 trades/year to avoid fee drag in choppy 2025 market.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_1d_camarilla_volume_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precomputed)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d Chop Index for regime filter (14-period)
    hl_range_1d = df_1d['high'] - df_1d['low']
    sum_range_14 = pd.Series(hl_range_1d).rolling(window=14, min_periods=14).sum().values
    abs_close_diff_14 = pd.Series(abs(df_1d['close'] - df_1d['close'].shift(1))).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_range_14 / abs_close_diff_14) / np.log10(14)
    chop[abs_close_diff_14 == 0] = 50  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Previous day's OHLC for Camarilla calculation
    prev_day_open = df_1d['open'].shift(1).values
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    rang = prev_day_high - prev_day_low
    H4 = prev_day_close + 1.5 * rang
    L4 = prev_day_close - 1.5 * rang
    H3 = prev_day_close + 1.1 * rang
    L3 = prev_day_close - 1.1 * rang
    
    # Align Camarilla levels to 1h
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_avg_20[i]) or not in_session[i]):
            # Hold previous position or go flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: chop > 50 = ranging (mean revert), chop < 50 = trending (breakout)
        is_ranging = chop_aligned[i] > 50
        is_trending = chop_aligned[i] < 50
        
        # Entry logic based on regime
        if is_ranging:
            # Ranging market: mean revert from H3/L3
            # Long when price crosses below L3 and shows rejection
            long_condition = (close[i] < L3_aligned[i] and 
                             close[i] > L4_aligned[i] and  # above L4 (support)
                             close[i-1] >= L3_aligned[i-1] and  # was above L3
                             vol_confirm[i])
            
            # Short when price crosses above H3 and shows rejection
            short_condition = (close[i] > H3_aligned[i] and
                              close[i] < H4_aligned[i] and  # below H4 (resistance)
                              close[i-1] <= H3_aligned[i-1] and  # was below H3
                              vol_confirm[i])
        else:
            # Trending market: breakout of H4/L4
            # Long when price breaks above H4 with volume
            long_condition = (close[i] > H4_aligned[i] and
                             close[i-1] <= H4_aligned[i-1] and  # was below or at H4
                             vol_confirm[i] and
                             close[i] > ema_50_4h_aligned[i])  # above 4h trend
            
            # Short when price breaks below L4 with volume
            short_condition = (close[i] < L4_aligned[i] and
                              close[i-1] >= L4_aligned[i-1] and  # was above or at L4
                              vol_confirm[i] and
                              close[i] < ema_50_4h_aligned[i])  # below 4h trend
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: price reverts to mean (H3) or stops below L4
            exit_long = (close[i] >= H3_aligned[i] or 
                        close[i] < L4_aligned[i])
        elif position == -1:
            # Exit short: price reverts to mean (L3) or stops above H4
            exit_short = (close[i] <= L3_aligned[i] or
                         close[i] > H4_aligned[i])
        
        # Generate signals
        if long_condition and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_condition and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals