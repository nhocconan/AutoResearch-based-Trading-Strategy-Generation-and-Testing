#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12/26 EMA crossover with daily volume confirmation and ADX trend filter.
# Uses EMA crossover for trend direction, daily volume to confirm momentum, and ADX to avoid choppy markets.
# Designed to work in both bull and bear markets by filtering for trending conditions only.
# Target: 80-120 total trades over 4 years (20-30/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12-period and 26-period EMA on daily
    close_1d = df_1d['close'].values
    ema_12 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Calculate daily ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 4-hour timeframe
    ema_12_aligned = align_htf_to_ltf(prices, df_1d, ema_12)
    ema_26_aligned = align_htf_to_ltf(prices, df_1d, ema_26)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_12_aligned[i]) or np.isnan(ema_26_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 1.5x daily volume MA (adjusted for 4h)
        # 6 4h periods per day, so daily MA/6 = approximate 4h period MA
        volume_4h_approx_ma = volume_ma_20_1d_aligned[i] / 6
        volume_condition = volume[i] > (volume_4h_approx_ma * 1.5)
        
        # ADX condition: trending market (ADX > 25)
        trend_condition = adx_aligned[i] > 25
        
        # EMA crossover conditions
        ema_fast_above = ema_12_aligned[i] > ema_26_aligned[i]
        ema_fast_below = ema_12_aligned[i] < ema_26_aligned[i]
        
        if position == 0:
            if ema_fast_above and volume_condition and trend_condition:
                position = 1
                signals[i] = position_size
            elif ema_fast_below and volume_condition and trend_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when EMA crossover turns bearish or ADX drops (trend weakening)
            if ema_fast_below or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when EMA crossover turns bullish or ADX drops (trend weakening)
            if ema_fast_above or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_EMA_Crossover_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0