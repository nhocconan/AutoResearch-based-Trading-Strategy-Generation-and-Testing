#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation
# Long when price breaks above Donchian upper AND 12h HMA21 slope positive AND volume > 2.0x 20 EMA
# Short when price breaks below Donchian lower AND 12h HMA21 slope negative AND volume > 2.0x 20 EMA
# Uses 4h for structure, 12h for trend to avoid whipsaw. Discrete sizing (0.25) to minimize fee churn.
# Target: 30-60 trades/year. Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_12hHMA_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get 12h data for HMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h HMA(21): HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # Pad arrays for WMA calculation
    wma_half = np.full_like(close_12h, np.nan)
    wma_full = np.full_like(close_12h, np.nan)
    
    for i in range(half_len, len(close_12h)):
        wma_half[i] = wma(close_12h[i-half_len+1:i+1], half_len)[-1] if i-half_len+1 >= 0 else np.nan
    
    for i in range(21, len(close_12h)):
        wma_full[i] = wma(close_12h[i-21+1:i+1], 21)[-1] if i-21+1 >= 0 else np.nan
    
    # HMA = WMA(2*WMA(half) - WMA(full), sqrt_len)
    hma_12h = np.full_like(close_12h, np.nan)
    for i in range(21, len(close_12h)):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2 * wma_half[i] - wma_full[i]
            # Need sqrt_len points for final WMA
            if i >= sqrt_len - 1:
                wma_diff = wma(close_12h[i-sqrt_len+1:i+1], sqrt_len)
                if len(wma_diff) > 0 and not np.isnan(wma_diff[-1]):
                    hma_12h[i] = wma_diff[-1]
    
    # Calculate HMA slope (trend direction)
    hma_slope = np.diff(hma_12h, prepend=hma_12h[0])
    # Uptrend when slope > 0, downtrend when slope < 0
    uptrend_12h = hma_slope > 0
    downtrend_12h = hma_slope < 0
    
    # Align 12h trend to 4h timeframe
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(uptrend_12h_aligned[i]) or np.isnan(downtrend_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 12h uptrend AND volume spike
            if (close[i] > donchian_upper[i] and 
                uptrend_12h_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND 12h downtrend AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  downtrend_12h_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR 12h trend changes to downtrend
            if (close[i] < donchian_lower[i] or 
                downtrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR 12h trend changes to uptrend
            if (close[i] > donchian_upper[i] or 
                uptrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals