#!/usr/bin/env python3
"""
6h_1d_1w_Adaptive_Channel_Breakout_v1
Hypothesis: On 6h timeframe, trade breakouts from a volatility-adjusted channel (ATR-based Donchian)
with 1d trend filter (EMA50) and 1w regime filter (ADX). Long when price breaks above upper band
in uptrend (1d EMA50 up, 1w ADX>25); short when breaks below lower band in downtrend.
Exit when price crosses the midline (average of upper/lower band). Uses volume confirmation
(1.5x 24-period average) to avoid false breakouts. Designed for low trade frequency (15-30/year)
by requiring multi-timeframe alignment and volatility filtering. Works in bull/bear via 1w ADX
regime filter that reduces false signals in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Adaptive_Channel_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D EMA(50) FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1W ADX(14) FOR REGIME FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smoothed_avg(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_1w = smoothed_avg(tr, 14)
    dm_plus_smooth = smoothed_avg(dm_plus, 14)
    dm_minus_smooth = smoothed_avg(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = smoothed_avg(dx, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === ATR-BASED DONCHIAN CHANNEL (20-period) ===
    atr_period = 20
    atr_values = np.zeros(n)
    atr_sum = 0.0
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_sum += tr
        if i >= atr_period:
            atr_sum -= max(high[i-atr_period] - low[i-atr_period], 
                          abs(high[i-atr_period] - close[i-atr_period-1]), 
                          abs(low[i-atr_period] - close[i-atr_period-1]))
        if i + 1 >= atr_period:
            atr_values[i] = atr_sum / atr_period
        else:
            atr_values[i] = np.nan
    
    # Donchian channels
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    hh_sum = 0
    ll_sum = 0
    hh_count = 0
    ll_count = 0
    for i in range(n):
        # Update highest high
        if i == 0:
            hh_sum = high[i]
            hh_count = 1
        else:
            hh_sum += high[i]
            hh_count += 1
            if i >= atr_period:
                hh_sum -= high[i-atr_period]
                hh_count -= 1
        if hh_count > 0:
            highest_high[i] = hh_sum / hh_count
        
        # Update lowest low
        if i == 0:
            ll_sum = low[i]
            ll_count = 1
        else:
            ll_sum += low[i]
            ll_count += 1
            if i >= atr_period:
                ll_sum -= low[i-atr_period]
                ll_count -= 1
        if ll_count > 0:
            lowest_low[i] = ll_sum / ll_count
    
    # Upper and lower bands
    upper_band = highest_high + atr_values
    lower_band = lowest_low - atr_values
    midline = (upper_band + lower_band) / 2
    
    # Volume average (24-period for 6h = ~6 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 24:
            vol_sum -= volume[i-24]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(midline[i]) or 
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: 1d EMA50 slope (approximated by current vs previous)
        ema_now = ema_50_1d_aligned[i]
        ema_prev = ema_50_1d_aligned[i-1] if i > 0 else ema_now
        ema_slope_up = ema_now > ema_prev
        ema_slope_down = ema_now < ema_prev
        
        # Regime filter: 1w ADX > 25 indicates trending market
        regime_filter = adx_1w_aligned[i] > 25
        
        # Entry conditions
        long_breakout = close[i] > upper_band[i]
        short_breakout = close[i] < lower_band[i]
        
        long_setup = long_breakout and vol_confirm and ema_slope_up and regime_filter
        short_setup = short_breakout and vol_confirm and ema_slope_down and regime_filter
        
        # Exit conditions: mean reversion to midline
        exit_long = close[i] < midline[i]
        exit_short = close[i] > midline[i]
        
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