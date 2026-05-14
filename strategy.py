#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (HMA21) and volume confirmation.
# Long when price breaks above Donchian upper band AND 1w HMA21 is rising (uptrend) AND 1w volume > 1.5 * 20-period average volume.
# Short when price breaks below Donchian lower band AND 1w HMA21 is falling (downtrend) AND 1w volume > 1.5 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 1d timeframe with strict entry conditions to avoid overtrading.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_Donchian20_Breakout_1wHMA21_1wVolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights, mode='valid') / weights.sum()
    
    wma_half = np.array([wma(arr[i:i+half_period], half_period) if i+half_period <= len(arr) else np.nan 
                         for i in range(len(arr) - half_period + 1)])
    wma_full = np.array([wma(arr[i:i+period], period) if i+period <= len(arr) else np.nan 
                         for i in range(len(arr) - period + 1)])
    
    raw_hma = 2 * wma_half - wma_full
    hma = np.array([wma(raw_hma[i:i+sqrt_period], sqrt_period) if i+sqrt_period <= len(raw_hma) else np.nan 
                    for i in range(len(raw_hma) - sqrt_period + 1)])
    
    # Pad to original length
    result = np.full_like(arr, np.nan)
    result[period-1:len(hma)+period-1] = hma
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w HMA21 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    hma_21_1w = calculate_hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Calculate 1w volume confirmation filter (HTF)
    volume_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1w = volume_1w > (1.5 * vol_ma_20_1w)  # Volume > 1.5x 20-period MA
    volume_confirm_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm_1w.astype(float))
    
    # Calculate Donchian(20) channels (based on prior 20 periods)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(20, n):
        period_high = np.max(high[i-20:i])
        period_low = np.min(low[i-20:i])
        donchian_upper[i] = period_high
        donchian_lower[i] = period_low
        donchian_mid[i] = (period_high + period_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_21_1w_aligned[i]) or 
            np.isnan(volume_confirm_1w_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper AND 1w HMA21 rising AND volume confirmation
            hma_rising = hma_21_1w_aligned[i] > hma_21_1w_aligned[i-1]
            if (open_[i] <= donchian_upper[i] and close[i] > donchian_upper[i] and 
                hma_rising and 
                volume_confirm_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower AND 1w HMA21 falling AND volume confirmation
            elif (open_[i] >= donchian_lower[i] and close[i] < donchian_lower[i] and 
                  not hma_rising and 
                  volume_confirm_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals