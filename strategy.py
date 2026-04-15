#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with volume confirmation and ADX trend filter
# Williams %R identifies overbought/oversold conditions. In trending markets (ADX > 25),
# we take mean-reversion entries when price pulls back from extremes. Volume confirms
# the strength of the pullback. Works in both bull and bear by only trading in the
# direction of the 4h trend (ADX > 25 and price > EMA50 for longs, price < EMA50 for shorts).
# Target: 80-180 total trades over 4 years (20-45/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for Williams %R and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / (tr14 + 1e-10)
    minus_di = 100 * minus_dm14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA50 on 4h for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: Williams %R oversold (< -80) + ADX > 25 (trending) + volume spike + price > EMA50
        if (williams_r_aligned[i] < -80 and
            adx_aligned[i] > 25 and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            close[i] > ema50_4h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R overbought (> -20) + ADX > 25 (trending) + volume spike + price < EMA50
        elif (williams_r_aligned[i] > -20 and
              adx_aligned[i] > 25 and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              close[i] < ema50_4h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or Williams %R returns to neutral range (-50 to -50)
        elif position == 1 and (williams_r_aligned[i] > -50 or adx_aligned[i] < 25):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (williams_r_aligned[i] < -50 or adx_aligned[i] < 25):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_ADX_Volume_Filter"
timeframe = "4h"
leverage = 1.0