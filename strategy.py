#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1w HMA trend filter
# Donchian breakout captures strong momentum moves in both bull/bear markets
# 1d volume > 2.0x 20-period average confirms institutional participation
# 1w HMA(21) trend filter: only trade in direction of weekly trend to avoid counter-trend whipsaws
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 75-200 trades over 4 years (19-50/year) with Sharpe > 0.5 on BTC/ETH

name = "4h_1d_1w_donchian_volume_hma_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d volume confirmation: 20-period average volume
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # 1w HMA(21) for trend filter
    close_1w = df_1w['close'].values
    hma_21_1w = calculate_hma(close_1w, 21)
    
    # Align HTF indicators to 4h timeframe (wait for bar close)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(hma_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x 1d average volume
        volume_confirmed = volume[i] > 2.0 * avg_volume_1d_aligned[i]
        
        # Trend filter: HMA slope > 0 = uptrend, < 0 = downtrend
        if i >= 101:  # Need previous value for slope
            hma_now = hma_21_1w_aligned[i]
            hma_prev = hma_21_1w_aligned[i-1]
            trend_up = hma_now > hma_prev
            trend_down = hma_now < hma_prev
        else:
            trend_up = False
            trend_down = False
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: breakout with volume confirmation in trend direction
            if close[i] > highest_high[i] and volume_confirmed and trend_up:
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and volume_confirmed and trend_down:
                position = -1
                signals[i] = -0.25
    
    return signals

def calculate_hma(values, period):
    """Calculate Hull Moving Average"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    
    def wma(data, window):
        if len(data) < window:
            return np.full(len(data), np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights, mode='valid') / weights.sum()
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(values, half_period)
    wma_full = wma(values, period)
    
    # Handle NaN values from WMA
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    # Pad with NaN to match original length
    result = np.full(len(values), np.nan)
    start_idx = period - 1
    end_idx = start_idx + len(hma)
    if end_idx <= len(values):
        result[start_idx:end_idx] = hma
    
    return result