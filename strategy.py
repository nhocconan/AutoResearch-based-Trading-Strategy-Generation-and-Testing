#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and chop regime filter
# In trending markets (1d CHOP < 42): breakout in direction of 1d EMA50 trend
# In ranging markets (1d CHOP >= 42): mean reversion at Donchian midpoint
# Volume confirmation (>1.8x 24-period EMA) filters low-quality breakouts
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.
# Strategy adapts to bull/bear markets via regime filter and uses 12h primary timeframe.

name = "12h_Donchian20_1dChop_Volume_EMA50"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filter and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Choppiness Index (CHOP) - 14 period
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d.sub(low_1d)
    tr2 = high_1d.sub(close_1d.shift(1)).abs()
    tr3 = low_1d.sub(close_1d.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over 14 periods
    tr_sum_14 = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh_14 = high_1d.rolling(window=14, min_periods=14).max()
    ll_14 = low_1d.rolling(window=14, min_periods=14).min()
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    hh_ll_diff = hh_14 - ll_14
    chop_1d = np.where(
        (hh_ll_diff > 0) & (~tr_sum_14.isna()) & (~hh_ll_diff.isna()),
        100 * np.log10(tr_sum_14 / hh_ll_diff) / np.log10(14),
        50.0  # neutral when undefined
    )
    
    # Align 1d indicators to 12h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h Donchian channels (20-period)
    # Use rolling window on 12h data directly
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: 24-period EMA of volume on 12h timeframe
    vol_ema_24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ema_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8 x 24-period EMA
        volume_confirm = volume[i] > (1.8 * vol_ema_24[i])
        
        if position == 0:
            if chop_aligned[i] < 42:
                # Trending market: breakout in direction of 1d EMA50 trend
                if close[i] > ema50_aligned[i]:
                    # Uptrend: long on break above Donchian high
                    if close[i] > donchian_high[i] and volume_confirm:
                        signals[i] = 0.25
                        position = 1
                else:
                    # Downtrend: short on break below Donchian low
                    if close[i] < donchian_low[i] and volume_confirm:
                        signals[i] = -0.25
                        position = -1
            else:
                # Ranging market: mean reversion at Donchian midpoint
                if close[i] <= donchian_low[i] and volume_confirm:
                    # Long at support
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donchian_high[i] and volume_confirm:
                    # Short at resistance
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR chop increases (>50) OR volume drops
            if (close[i] >= donchian_mid[i] or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_24[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR chop increases (>50) OR volume drops
            if (close[i] <= donchian_mid[i] or 
                chop_aligned[i] > 50 or 
                volume[i] < vol_ema_24[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals