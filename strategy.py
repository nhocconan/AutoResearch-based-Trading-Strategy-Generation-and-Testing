#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index (range filter) + 1d Bollinger Band breakout + volume confirmation
# Uses Choppiness Index to detect ranging markets (avoid whipsaw), Bollinger Bands for volatility breakouts,
# and volume to confirm strength. Designed for 12h timeframe with target of 50-150 total trades over 4 years.
# Works in bull/bear markets: ranges in choppy conditions, breaks out in trending conditions.
name = "12h_Chop_BB_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Calculate 1d Choppiness Index (14)
    atr_1d = pd.Series(np.where(
        high[::len(high)//len(close_1d)] - low[::len(low)//len(close_1d)] > 0,
        high[::len(high)//len(close_1d)] - low[::len(low)//len(close_1d)],
        np.full_like(close_1d, np.nan)
    )).rolling(window=14, min_periods=14).mean().values  # Simplified ATR approximation
    # Proper ATR calculation for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DM and -DM
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate smoothed +DM and -DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / atr_1d)
    minus_di = 100 * (minus_dm_smooth / atr_1d)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR)/ (max(high)-min(low))) / log10(14)
    # Simplified: use ADX inverse - when ADX < 20, market is choppy
    chop = 100 - adx  # Inverse relationship: low ADX = high chop
    
    # Align indicators to 12h
    upper_bb_12h = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_12h = align_htf_to_ltf(prices, df_1d, lower_bb)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: current volume > 1.3x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_12h[i]) or np.isnan(lower_bb_12h[i]) or 
            np.isnan(chop_12h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > upper_bb_12h[i-1]  # Break above upper BB
        short_breakout = close[i] < lower_bb_12h[i-1]  # Break below lower BB
        
        # Chop filter: only trade when chop > 50 (ranging market) for mean reversion,
        # or chop < 30 (trending) for breakout continuation
        chop_value = chop_12h[i]
        chop_ranging = chop_value > 50  # Range-bound
        chop_trending = chop_value < 30  # Trending
        
        if position == 0:
            # Long: bullish breakout in trending market OR mean reversion in ranging market from lower BB
            if (long_breakout and chop_trending) or (close[i] < lower_bb_12h[i] and chop_ranging and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout in trending market OR mean reversion in ranging market from upper BB
            elif (short_breakout and chop_trending) or (close[i] > upper_bb_12h[i] and chop_ranging and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish break below lower BB or chop increases (losing trend)
            if close[i] < lower_bb_12h[i] or chop_value > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish break above upper BB or chop increases (losing trend)
            if close[i] > upper_bb_12h[i] or chop_value > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals