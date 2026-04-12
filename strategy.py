#!/usr/bin/env python3
"""
6h_1w_1d_Camarilla_Reverse_Signal
Hypothesis: On 6h timeframe, fade daily C3/C4 levels when weekly trend is strong, and breakout when weekly trend is weak.
Use weekly ADX to determine regime: ADX>25 = trend (use daily C3/C4 as breakout), ADX<25 = range (fade at daily C3/C4).
Requires volume confirmation and 60-minute time filter to avoid low-activity periods.
Designed to work in both bull and bear markets by adapting to regime.
Target: 80-160 total trades over 4 years (20-40/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Camarilla_Reverse_Signal"
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
    
    # === WEEKLY ADX FOR REGIME ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Wilder's ADX (14)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1]) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            minus_dm[i] = max(0, low[i-1] - low[i]) if low[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.nansum(tr[:period]) if not np.any(np.isnan(tr[:period])) else np.nan
        plus_dm_sum = np.nansum(plus_dm[:period]) if not np.any(np.isnan(plus_dm[:period])) else np.nan
        minus_dm_sum = np.nansum(minus_dm[:period]) if not np.any(np.isnan(minus_dm[:period])) else np.nan
        
        if np.isnan(atr[period-1]) or atr[period-1] == 0:
            return np.full_like(high, np.nan)
            
        atr[period-1:] = (atr[period-2:-1] * (period-1) + tr[period-1:]) / period
        plus_dm_sum[period-1:] = (plus_dm_sum[period-2:-1] * (period-1) + plus_dm[period-1:]) / period
        minus_dm_sum[period-1:] = (minus_dm_sum[period-2:-1] * (period-1) + minus_dm[period-1:]) / period
        
        plus_di = 100 * plus_dm_sum / atr
        minus_di = 100 * minus_dm_sum / atr
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        adx = np.full_like(high, np.nan)
        adx[2*period-2:] = np.nan
        dx_valid = dx[2*period-1:]
        if len(dx_valid) > 0:
            adx[2*period-1] = np.nanmean(dx_valid[:period]) if not np.any(np.isnan(dx_valid[:period])) else np.nan
            for i in range(2*period, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_6h = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === DAILY CAMARILLA LEVELS (C3, C4) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_c3 = np.full(len(close_1d), np.nan)
    camarilla_c4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        range_val = high_1d[i] - low_1d[i]
        camarilla_c3[i] = close_1d[i] + range_val * 1.1 / 4  # C3
        camarilla_c4[i] = close_1d[i] + range_val * 1.1 / 2  # C4
    
    c3_6h = align_htf_to_ltf(prices, df_1d, camarilla_c3)
    c4_6h = align_htf_to_ltf(prices, df_1d, camarilla_c4)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === TIME FILTER (08-20 UTC) ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready or outside time filter
        if (np.isnan(adx_1w_6h[i]) or np.isnan(c3_6h[i]) or np.isnan(c4_6h[i]) or 
            np.isnan(vol_ratio[i]) or not time_filter[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime determination
        trending = adx_1w_6h[i] > 25
        ranging = adx_1w_6h[i] < 25
        
        # Signals based on regime
        if trending:
            # Breakout mode: break above C4 = long, break below C3 = short
            long_signal = close[i] > c4_6h[i] * 1.001 and vol_ratio[i] > 1.5
            short_signal = close[i] < c3_6h[i] * 0.999 and vol_ratio[i] > 1.5
        else:
            # Reverse mode: fade at C3/C4
            long_signal = close[i] < c3_6h[i] * 0.999 and vol_ratio[i] > 1.5  # Oversold bounce
            short_signal = close[i] > c4_6h[i] * 1.001 and vol_ratio[i] > 1.5  # Overbought rejection
        
        # Exit conditions
        exit_long = position == 1 and (
            (trending and close[i] < c3_6h[i]) or  # Breakdown in trend
            (not trending and close[i] > c4_6h[i] * 0.999)  # Reversal in range
        )
        exit_short = position == -1 and (
            (trending and close[i] > c4_6h[i]) or  # Breakout in trend
            (not trending and close[i] < c3_6h[i] * 1.001)  # Reversal in range
        )
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals