#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d trend alignment and volatility regime filter.
# Uses 1d EMA34 for trend direction and 12h Donchian breakout for entry timing.
# Enters only during 08-20 UTC session to avoid low-volume noise.
# Uses Choppiness Index (14) > 61.8 to filter for ranging markets where mean reversion works.
# Targets 15-37 trades/year (60-150 total over 4 years) with strict entry conditions.
# Works in bull/bear by following higher timeframe trends and avoiding choppy whipsaws.
name = "12h_1d_EMA34_Donchian20_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for Donchian20 breakout (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    # Donchian channels: 20-period high/low
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    high_20_12h_aligned = align_htf_to_ltf(prices, df_12h, high_20_12h)
    low_20_12h_aligned = align_htf_to_ltf(prices, df_12h, low_20_12h)
    
    # Choppiness Index (14) on 12h data - high values indicate ranging market
    # CHOP = 100 * log10(sum(ATR over n) / (log10(highest-high - lowest-low) * n)) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10((atr_14 * 14) / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_filter = chop_raw > 61.8  # ranging market filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_20_12h_aligned[i]) or 
            np.isnan(low_20_12h_aligned[i]) or np.isnan(chop_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d EMA34 AND breaks 12h Donchian high in ranging market
            if (close[i] > ema_34_1d_aligned[i] and 
                close[i] > high_20_12h_aligned[i] and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA34 AND breaks 12h Donchian low in ranging market
            elif (close[i] < ema_34_1d_aligned[i] and 
                  close[i] < low_20_12h_aligned[i] and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d EMA34 or 12h Donchian low
            if close[i] < ema_34_1d_aligned[i] or close[i] < low_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d EMA34 or 12h Donchian high
            if close[i] > ema_34_1d_aligned[i] or close[i] > high_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals