#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Choppiness Index regime filter with 1-week Donchian breakout
# Long when weekly price breaks above Donchian(20) high + daily Choppiness Index > 61.8 (ranging market)
# Short when weekly price breaks below Donchian(20) low + daily Choppiness Index > 61.8
# Exit when price crosses 10-day SMA in opposite direction
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily Choppiness Index to identify ranging markets where mean reversion works
# Weekly Donchian breakouts provide directional bias in ranging conditions
# Target: 50-120 total trades over 4 years (12-30/year) - suitable for 1d timeframe

name = "1d_chop_ranging_weekly_donchian_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-week data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14d = hh_14d - ll_14d
    range_14d = np.where(range_14d == 0, 1e-10, range_14d)
    
    # Choppiness Index: 100 * log10(tr_sum / range_14d) / log10(14)
    chop_ratio = tr_sum / range_14d
    chop_ratio = np.where(chop_ratio <= 0, 1e-10, chop_ratio)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1-week Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    highest_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    highest_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low)
    
    # 10-day SMA for exit
    sma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_high_aligned[i]) or 
            np.isnan(lowest_low_aligned[i]) or np.isnan(sma_10[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 10-day SMA
            elif close[i] < sma_10[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 10-day SMA
            elif close[i] > sma_10[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout in ranging market (Choppiness > 61.8)
            chop_filter = chop_aligned[i] > 61.8  # Ranging market condition
            
            # Long: price breaks above weekly Donchian high + ranging market
            if close[i] > highest_high_aligned[i] and chop_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below weekly Donchian low + ranging market
            elif close[i] < lowest_low_aligned[i] and chop_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals