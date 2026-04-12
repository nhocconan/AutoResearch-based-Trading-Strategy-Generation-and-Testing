#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Breakout_Volume_Regime_v1
Hypothesis: On 12h timeframe, enter long at Camarilla L3 with 1d uptrend and volume spike, short at H3 with 1d downtrend and volume spike.
Exit at L4/H4 or opposite Camarilla level. Uses weekly ADX to filter for trending markets only.
Designed for low trade frequency (<30/year) by requiring confluence of Camarilla level, trend, volume, and regime.
Works in bull/bear via 1d trend filter and avoids choppy markets via weekly ADX < 25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camarilla_Breakout_Volume_Regime_v1"
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
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day)
    range_1d = high_1d - low_1d
    close_prev = close_1d  # same day close for intraday
    
    # Camarilla levels for intraday trading
    h4 = close_prev + 1.1 * range_1d / 2
    h3 = close_prev + 1.1 * range_1d / 4
    h2 = close_prev + 1.1 * range_1d / 6
    h1 = close_prev + 1.1 * range_1d / 12
    l1 = close_prev - 1.1 * range_1d / 12
    l2 = close_prev - 1.1 * range_1d / 6
    l3 = close_prev - 1.1 * range_1d / 4
    l4 = close_prev - 1.1 * range_1d / 2
    
    # === 1D EMA(50) FOR TREND FILTER ===
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_1d = np.full_like(close_1d, np.nan)
    
    # === WEEKLY ADX(14) FOR REGIME FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(arr[1:period]) 
        # Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1w = smooth_wilder(tr, 14)
    dm_plus_smooth = smooth_wilder(dm_plus, 14)
    dm_minus_smooth = smooth_wilder(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr_1w > 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w > 0, 100 * dm_minus_smooth / atr_1w, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = smooth_wilder(dx, 14)
    
    # Volume average (12-period for 12h = ~6 days)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 12:
            vol_sum -= volume[i-12]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    # Align data to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending = adx_1w_aligned[i] > 25
        
        # Volume confirmation: at least 2x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Trend filter: price above/below 1d EMA(50)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_setup = (close[i] > h3_aligned[i]) and trending and vol_confirm and price_above_ema
        short_setup = (close[i] < l3_aligned[i]) and trending and vol_confirm and price_below_ema
        
        # Exit conditions: mean reversion to opposite level or stop at L4/H4
        exit_long = close[i] < l4_aligned[i] or close[i] < l3_aligned[i]
        exit_short = close[i] > h4_aligned[i] or close[i] > h3_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals