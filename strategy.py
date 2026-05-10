#!/usr/bin/env python3
# 1h_4h1d_Camarilla_R1_S1_Breakout_Volume_Filter
# Hypothesis: 1h strategy using 4h trend (price > 4h EMA50) and 1d regime (ADX < 25 for range) as filters,
# entering on 1h Camarilla R1/S1 breakouts with volume confirmation (1.5x 20-bar average).
# Exits on close crossing 1h EMA10 in opposite direction.
# Designed for low trade frequency (15-35/year) to minimize fee impact in ranging and trending markets.

name = "1h_4h1d_Camarilla_R1_S1_Breakout_Volume_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for regime filter (ADX) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d ADX for regime filter (range: ADX < 25)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smoothed(series, period):
        result = np.full_like(series, np.nan)
        if len(series) < period:
            return result
        result[period-1] = np.nansum(series[:period])
        for i in range(period, len(series)):
            result[i] = result[i-1] - (result[i-1] / period) + series[i]
        return result
    
    atr = smoothed(tr, 14)
    dm_plus_smooth = smoothed(dm_plus, 14)
    dm_minus_smooth = smoothed(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed(dx, 14)
    
    # Align ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d Camarilla levels (R1, S1) from previous day
    r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align R1 and S1 to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1h EMA10 for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_10[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: range market (ADX < 25)
        range_regime = adx_aligned[i] < 25
        
        # Trend filter: price above/below 4h EMA50
        price_above_4h_ema = close[i] > ema_50_4h_aligned[i]
        price_below_4h_ema = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above R1 in range regime with volume
            if close[i] > r1_aligned[i] and range_regime and volume_surge:
                signals[i] = 0.20
                position = 1
            # Short: breakdown below S1 in range regime with volume
            elif close[i] < s1_aligned[i] and range_regime and volume_surge:
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Long exit: price crosses below EMA10
                if close[i] < ema_10[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Short exit: price crosses above EMA10
                if close[i] > ema_10[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals