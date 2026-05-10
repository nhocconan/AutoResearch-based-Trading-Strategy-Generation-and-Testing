#!/usr/bin/env python3
# 6h_ADX_Trend_RSI_Momentum
# Hypothesis: Combines ADX (trend strength) with RSI momentum on 6h timeframe, using 1d trend filter to avoid counter-trend trades.
# ADX > 25 indicates strong trend; RSI > 55 for long, < 45 for short; 1d EMA50 filters trend direction.
# Designed for 6h to achieve 12-37 trades/year, suitable for both bull and bear markets by following established trends.

name = "6h_ADX_Trend_RSI_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value has no previous close
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) >= period:
                # First value is simple average
                result[period-1] = np.nansum(arr[:period])
                # Subsequent values: Wilder smoothing
                for i in range(period, len(arr)):
                    result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result
        
        tr_smooth = smooth_wilder(tr, period)
        dm_plus_smooth = smooth_wilder(dm_plus, period)
        dm_minus_smooth = smooth_wilder(dm_minus, period)
        
        # Directional Indicators
        plus_di = np.where(tr_smooth != 0, (dm_plus_smooth / tr_smooth) * 100, 0)
        minus_di = np.where(tr_smooth != 0, (dm_minus_smooth / tr_smooth) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = smooth_wilder(dx, period)
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # RSI (14-period)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)  # First value has no delta
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) >= period:
                result[period-1] = np.nansum(arr[:period])
                for i in range(period, len(arr)):
                    result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result
        
        avg_gain = smooth_wilder(gain, period)
        avg_loss = smooth_wilder(loss, period)
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (strong trend), RSI > 55 (bullish momentum), price above 1d EMA50
            if adx[i] > 25 and rsi[i] > 55 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend), RSI < 45 (bearish momentum), price below 1d EMA50
            elif adx[i] > 25 and rsi[i] < 45 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ADX < 20 (weakening trend) or RSI < 50 (loss of momentum) or price below 1d EMA50
            if adx[i] < 20 or rsi[i] < 50 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX < 20 (weakening trend) or RSI > 50 (loss of momentum) or price above 1d EMA50
            if adx[i] < 20 or rsi[i] > 50 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals