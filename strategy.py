#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 20-day Donchian breakout with volume confirmation and ADX trend filter.
# Long when price breaks above 20-day high + volume > average + ADX > 25.
# Short when price breaks below 20-day low + volume > average + ADX > 25.
# Exit when price crosses the 20-day midpoint.
# Uses weekly trend filter: only trade in direction of weekly 50-period EMA.
# Designed for fewer, high-quality trades to avoid fee drag and work in both bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-day Donchian channels
    highest_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    mid_20d = (highest_20d + lowest_20d) / 2.0
    
    # Align daily Donchian levels to 1d timeframe (no alignment needed as prices are 1d)
    highest_20d_aligned = highest_20d
    lowest_20d_aligned = lowest_20d
    mid_20d_aligned = mid_20d
    
    # ADX calculation (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Volume filter: 1d volume > 20-day average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(highest_20d_aligned[i]) or np.isnan(lowest_20d_aligned[i]) or np.isnan(adx[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: only long if above weekly EMA50, only short if below
        weekly_trend_up = close[i] > ema_50_1w_aligned[i]
        weekly_trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # ADX trend filter: only trade when ADX > 25
        trend_filter = adx[i] > 25
        
        price = close[i]
        upper_band = highest_20d_aligned[i]
        lower_band = lowest_20d_aligned[i]
        mid_point = mid_20d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above 20-day high + volume + ADX + weekly trend up
            if price > upper_band and vol_filter and trend_filter and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-day low + volume + ADX + weekly trend down
            elif price < lower_band and vol_filter and trend_filter and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 20-day midpoint
            if price < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-day midpoint
            if price > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_TrendFilter_Volume_ADX"
timeframe = "1d"
leverage = 1.0