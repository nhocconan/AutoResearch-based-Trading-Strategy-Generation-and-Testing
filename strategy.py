#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and ATR(14) volatility filter.
- Primary timeframe: 4h for execution, HTF: 12h for HMA trend and Donchian channels.
- Donchian channels from prior 12h: upper = max(high,20), lower = min(low,20) 
  Long when price breaks above upper channel with volatility expansion, Short when breaks below lower.
- Trend filter: Only trade in direction of 12h HMA21 (long if HMA21 rising, short if falling).
- Volatility filter: ATR(14) > 0.5 * 50-period ATR MA to ensure sufficient momentum.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data for HMA21 trend filter and Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) from prior 12h bar
    # Upper = max(high,20), Lower = min(low,20)
    def rolling_max(arr, window):
        return np.convolve(arr, np.ones(window), 'valid')[:len(arr)-window+1] if len(arr) >= window else np.full(len(arr), np.nan)
    def rolling_min(arr, window):
        return np.convolve(arr, np.ones(window), 'valid')[:len(arr)-window+1] if len(arr) >= window else np.full(len(arr), np.nan)
    
    # Proper rolling max/min using pandas for simplicity and correctness
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h (each 12h bar = 3x 4h bars)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # 12h HMA21 for trend filter: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def hma(values, window):
        half = window // 2
        sqrt_n = int(np.sqrt(window))
        if half == 0:
            return np.full_like(values, np.nan)
        wma_half = np.array([np.nan] * half + list(wma(values, half)))
        wma_full = np.array([np.nan] * window + list(wma(values, window)))
        raw_hma = 2 * wma_half - wma_full
        hma_values = np.array([np.nan] * (sqrt_n - 1) + list(wma(raw_hma[~np.isnan(raw_hma)], sqrt_n)) if len(raw_hma[~np.isnan(raw_hma)]) >= sqrt_n else np.full_like(raw_hma, np.nan))
        # Align back to original length
        result = np.full_like(values, np.nan)
        start_idx = len(values) - len(hma_values)
        if start_idx >= 0:
            result[start_idx:] = hma_values
        return result
    
    hma_21_12h = hma(close_12h, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # ATR(14) volatility filter on 12h
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev_12h = np.roll(close_12h, 1)
    close_prev_12h[0] = np.nan
    tr_12h = true_range(high_12h, low_12h, close_prev_12h)
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_ma_50_12h = pd.Series(atr_14_12h).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_14_12h > (0.5 * atr_ma_50_12h)
    vol_filter_aligned = align_htf_to_ltf(prices, df_12h, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Donchian20 + HMA21 + ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 12h HMA21 trend
            if i > 0 and not np.isnan(hma_21_12h_aligned[i-1]):
                hma_slope = hma_21_12h_aligned[i] - hma_21_12h_aligned[i-1]
                if hma_slope > 0:  # Uptrend
                    # Long when price breaks above upper Donchian with volatility expansion
                    if close[i] > donchian_upper_aligned[i] and vol_filter_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                elif hma_slope < 0:  # Downtrend
                    # Short when price breaks below lower Donchian with volatility expansion
                    if close[i] < donchian_lower_aligned[i] and vol_filter_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian or opposite signal
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian or opposite signal
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_HMA21_Trend_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0