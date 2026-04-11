#!/usr/bin/env python3
# 6h_12h_1d_market_regime_v1
# Strategy: 6h market regime with 12h/1d trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Uses 12h ADX to identify trending vs ranging markets and 1d EMA for directional bias.
# In trending markets (ADX > 25): enter breakouts in direction of 1d EMA trend.
# In ranging markets (ADX <= 25): fade at Bollinger Bands extremes.
# Designed for low trade frequency (~20-40/year) to minimize fee drift.
# Works in bull markets via trend-following breakouts and bear markets via mean reversion.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_market_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Load 1d data ONCE before loop for EMA and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h ADX calculation (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smooth_wmma(arr, period):
        """Wilder's moving average (smoothed)"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period]) / period
        # Subsequent values
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_12h = smooth_wmma(tr, 14)
    dm_plus_smooth = smooth_wmma(dm_plus, 14)
    dm_minus_smooth = smooth_wmma(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h != 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h != 0, 100 * dm_minus_smooth / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_12h = smooth_wmma(dx, 14)
    
    # Align 12h ADX to 6s timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 1d EMA50 for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(adx_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(sma_20[i]) or np.isnan(std_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        adx = adx_12h_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        curr_close = close[i]
        
        # Regime determination
        is_trending = adx > 25
        is_ranging = adx <= 25
        
        if is_trending:
            # Trending market: breakout in direction of 1d EMA
            if curr_close > ema_50:  # Bullish bias
                # Long breakout: price above upper Bollinger Band
                if curr_close > upper and position != 1:
                    position = 1
                    signals[i] = 0.25
                # Exit: price crosses below EMA
                elif position == 1 and curr_close < ema_50:
                    position = 0
                    signals[i] = 0.0
            else:  # Bearish bias
                # Short breakdown: price below lower Bollinger Band
                if curr_close < lower and position != -1:
                    position = -1
                    signals[i] = -0.25
                # Exit: price crosses above EMA
                elif position == -1 and curr_close > ema_50:
                    position = 0
                    signals[i] = 0.0
        else:
            # Ranging market: mean reversion at Bollinger Bands
            # Long: price touches lower BB and starts to revert up
            if curr_close < lower and i > 50 and close[i-1] >= lower and position != 1:
                position = 1
                signals[i] = 0.25
            # Short: price touches upper BB and starts to revert down
            elif curr_close > upper and i > 50 and close[i-1] <= upper and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit: price returns to middle (SMA)
            elif position == 1 and curr_close > sma_20[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and curr_close < sma_20[i]:
                position = 0
                signals[i] = 0.0
        
        # Hold position
        if position == 1 and signals[i] == 0:
            signals[i] = 0.25
        elif position == -1 and signals[i] == 0:
            signals[i] = -0.25
    
    return signals