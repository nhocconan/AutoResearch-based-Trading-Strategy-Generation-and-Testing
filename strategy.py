#!/usr/bin/env python3
# 6h_adx_williams_alligator_v1
# Hypothesis: 6h strategy combining ADX trend strength with Williams Alligator (Jaw/Teeth/Lips) for trend confirmation. 
# Long when: ADX > 25 (strong trend) + price > Alligator Teeth (mid) + Alligator Lips > Jaw (bullish alignment)
# Short when: ADX > 25 + price < Alligator Teeth + Alligator Lips < Jaw (bearish alignment)
# Uses 1d HTF Donchian(20) for breakout alignment to avoid counter-trend trades.
# Discrete sizing (0.25) to limit fee churn. Target: 12-37 trades/year (50-150 total over 4 years).
# Works in bull/bear: ADX filters weak trends, Alligator provides dynamic S/R, HTF Donchian ensures structure alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_williams_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator (13,8,5) - smoothed with SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Jaw (13-period SMMA of high)
    teeth = smma(low, 8)   # Teeth (8-period SMMA of low)
    lips = smma(close, 5)  # Lips (5-period SMMA of close)
    
    # ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX, +DI, -DI"""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
        tr_series = pd.Series(tr)
        plus_dm_series = pd.Series(plus_dm)
        minus_dm_series = pd.Series(minus_dm)
        
        atr = tr_series.ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        plus_di = 100 * (plus_dm_series.ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr)
        minus_di = 100 * (minus_dm_series.ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Multi-timeframe: 1d Donchian(20) for trend alignment
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    donchian_high_1d = high_1d_s.rolling(window=20, min_periods=20).max().values
    donchian_low_1d = low_1d_s.rolling(window=20, min_periods=20).min().values
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx[i]) or np.isnan(close[i]) or
            np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # ADX trend strength filter
        strong_trend = adx[i] > 25
        
        # HTF Donchian breakout alignment
        htf_uptrend = close[i] > donchian_high_1d_aligned[i]
        htf_downtrend = close[i] < donchian_low_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks bearish OR ADX weakens
            if not bullish_alignment or adx[i] < 20:  # Exit if alignment lost or trend weakens
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks bullish OR ADX weakens
            if not bearish_alignment or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish alignment + strong trend + HTF uptrend alignment
            if bullish_alignment and strong_trend and htf_uptrend:
                position = 1
                signals[i] = 0.25
            # Enter short: bearish alignment + strong trend + HTF downtrend alignment
            elif bearish_alignment and strong_trend and htf_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals