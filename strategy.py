#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Donchian upper AND 1d ADX > 25 AND volume > 1.5x average
# Short when price breaks below Donchian lower AND 1d ADX > 25 AND volume > 1.5x average
# Exit when price crosses Donchian middle (mean reversion) OR ADX < 20 (trend weakness)
# Uses 4h timeframe with daily trend filter for BTC/ETH resilience in both bull and bear markets.
# Donchian provides structure, ADX filters choppy regimes, volume confirms breakout strength.

name = "4h_Donchian20_1dADX_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 4h data for Donchian calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels on 4h data (using previous bar's OHLC to avoid look-ahead)
    if len(high_4h) >= 20:
        # Rolling window of 20 periods on 4h data
        upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
        lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
        middle_4h = (upper_4h + lower_4h) / 2.0
    else:
        upper_4h = np.full_like(high_4h, np.nan)
        lower_4h = np.full_like(low_4h, np.nan)
        middle_4h = np.full_like(high_4h, np.nan)
    
    # Align Donchian levels to 4h timeframe (already aligned since calculated on 4h)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data for trend filter
    if len(high_1d) >= 14:
        # True Range
        tr1 = pd.Series(high_1d).diff().abs()
        tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift(1)).abs()
        tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        
        # Directional Movement
        up_move = pd.Series(high_1d).diff()
        down_move = -pd.Series(low_1d).diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    else:
        adx = np.full_like(high_1d, np.nan)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current 4h volume > 1.5x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for ADX and volume
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > upper AND 1d ADX > 25 (trending) AND volume spike
            if close[i] > upper_aligned[i] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < lower AND 1d ADX > 25 (trending) AND volume spike
            elif close[i] < lower_aligned[i] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < middle (mean reversion) OR ADX < 20 (trend weakness)
            if close[i] < middle_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > middle (mean reversion) OR ADX < 20 (trend weakness)
            if close[i] > middle_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals