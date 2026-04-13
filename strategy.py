#!/usr/bin/env python3
"""
1d_1W_KAMA_Trend_with_Volume_Confirmation
Hypothesis: Daily KAMA (adaptive trend) with weekly volume confirmation and ADX filter.
Long when KAMA indicates uptrend (close > KAMA) + weekly volume > 1.5x 20-week avg + weekly ADX > 25.
Short when KAMA indicates downtrend (close < KAMA) + weekly volume > 1.5x 20-week avg + weekly ADX > 25.
Exit when close crosses KAMA or weekly ADX < 20.
Designed for 1d timeframe to target 10-25 trades/year with adaptive trend following in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - 10-period
    def kama(close_prices, length=10):
        # Efficiency Ratio
        change = np.abs(np.diff(close_prices, n=length))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        # Fix: volatility calculation needs to be rolling sum of absolute changes
        volatility = np.array([np.sum(np.abs(np.diff(close_prices[i:i+length+1]))) 
                              if i+length < len(close_prices) else np.nan 
                              for i in range(len(close_prices))])
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
        # Initialize KAMA
        kama_vals = np.full_like(close_prices, np.nan, dtype=float)
        kama_vals[length] = close_prices[length]
        for i in range(length+1, len(close_prices)):
            if not np.isnan(kama_vals[i-1]) and not np.isnan(sc[i]):
                kama_vals[i] = kama_vals[i-1] + sc[i] * (close_prices[i] - kama_vals[i-1])
            else:
                kama_vals[i] = kama_vals[i-1]
        return kama_vals
    
    kama_vals = kama(close, 10)
    
    # Get weekly data for volume and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly volume average (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    
    # Weekly ADX calculation (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period])
        # Subsequent values: smoothed = prev - (prev/period) + current
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1]/period) + arr[i]
        return result
    
    tr_14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align weekly indicators to daily
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_vals[i]) or np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current weekly volume > 1.5x 20-period average
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w)
        vol_condition = vol_1w_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # ADX condition: trending market
        adx_condition = adx_aligned[i] > 25
        
        # KAMA trend conditions
        kama_uptrend = close[i] > kama_vals[i]
        kama_downtrend = close[i] < kama_vals[i]
        
        # Exit conditions
        exit_condition = (close[i] < kama_vals[i] if position == 1 else 
                         close[i] > kama_vals[i] if position == -1 else False)
        trend_weak = adx_aligned[i] < 20
        
        if position == 0:
            if kama_uptrend and vol_condition and adx_condition:
                position = 1
                signals[i] = position_size
            elif kama_downtrend and vol_condition and adx_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if exit_condition or trend_weak:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if exit_condition or trend_weak:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1W_KAMA_Trend_with_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0