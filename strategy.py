#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d HMA(21) trend + volume spike confirmation
# Long when: price breaks above 12h Donchian(20) high AND 1d HMA(21) shows uptrend AND volume > 2x 20-period MA
# Short when: price breaks below 12h Donchian(20) low AND 1d HMA(21) shows downtrend AND volume > 2x 20-period MA
# Exit when: price returns to 12h Donchian(20) midpoint OR opposite breakout occurs
# Uses Donchian for structure, HMA for trend, volume for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1dHMA21_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 12h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian(20) on 12h
    if len(high) >= 20 and len(low) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (highest_high + lowest_low) / 2.0
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Donchian breakout signals
    donchian_breakout_up = (close > highest_high) & (np.roll(close, 1) <= np.roll(highest_high, 1))
    donchian_breakout_down = (close < lowest_low) & (np.roll(close, 1) >= np.roll(lowest_low, 1))
    donchian_revert_mid = np.abs(close - donchian_mid) < 0.001 * close  # approximate midpoint return
    
    # Get 1d data ONCE before loop for HMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # need at least 21 periods for HMA
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d close
    close_1d = df_1d['close'].values
    if len(close_1d) >= 21:
        # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        def wma(arr, window):
            if len(arr) < window:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_1d, half_n)
        wma_full = wma(close_1d, 21)
        if len(wma_half) > 0 and len(wma_full) > 0:
            hma_raw = 2 * wma_half - wma_full
            hma_21 = wma(hma_raw, sqrt_n)
            # Pad to original length
            hma_21_padded = np.full(len(close_1d), np.nan)
            start_idx = len(close_1d) - len(hma_21)
            hma_21_padded[start_idx:] = hma_21
            hma_trend = hma_21_padded > np.roll(hma_21_padded, 1)  # upward slope
        else:
            hma_trend = np.zeros(len(close_1d), dtype=bool)
    else:
        hma_trend = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d HMA trend to 12h timeframe
    hma_trend_aligned = align_htf_to_ltf(prices, df_1d, hma_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_trend_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + HMA uptrend + volume filter
            if (donchian_breakout_up[i] and 
                hma_trend_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + HMA downtrend + volume filter
            elif (donchian_breakout_down[i] and 
                  hma_trend_aligned[i] == 0.0 and  # downward slope
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR short breakout occurs
            if (donchian_revert_mid[i] or donchian_breakout_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR long breakout occurs
            if (donchian_revert_mid[i] or donchian_breakout_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals