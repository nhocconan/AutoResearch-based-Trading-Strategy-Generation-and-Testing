#!/usr/bin/env python3
"""
1d Weekly Range Breakout with Volume Spike and ADX Trend Filter
Long: Price breaks above weekly high (Friday close) + volume > 2.0x 20-day avg + ADX(14) > 25
Short: Price breaks below weekly low (Friday close) + volume > 2.0x 20-day avg + ADX(14) > 25
Exit: Opposite break of weekly level
Target: 15-25 trades/year per symbol (30-100 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for weekly high/low (using Friday's close as weekly level)
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high']  # Weekly high
    weekly_low = df_1w['low']    # Weekly low
    
    # Align weekly levels to daily (wait for weekly close)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high.values)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low.values)
    
    # Volume confirmation: 20-day average volume
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) for trend strength filter
    # Calculate +DI, -DI, and DX
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_series - high_series.shift(1)
    down_move = low_series.shift(1) - low_series
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx_values[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_values[i]
        
        if position == 0:
            # Long: break above weekly high + volume spike + strong trend
            if price > weekly_high_aligned[i] and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low + volume spike + strong trend
            elif price < weekly_low_aligned[i] and vol > 2.0 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below weekly low
            if price < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above weekly high
            if price > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyRange_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0