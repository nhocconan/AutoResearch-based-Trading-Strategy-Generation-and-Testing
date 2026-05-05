#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
# Long when: price breaks above 1d Donchian(20) high AND 1w HMA is rising AND volume > 1.5x 20-period MA
# Short when: price breaks below 1d Donchian(20) low AND 1w HMA is falling AND volume > 1.5x 20-period MA
# Exit when: price returns to 1d Donchian(20) midpoint
# Uses Donchian for structure, weekly HMA for bias, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian20_1wHMA21_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 1d using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian(20) on 1d
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
    
    # Get 1w data ONCE before loop for HMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate HMA(21) on 1w close
    close_1w = df_1w['close'].values
    if len(close_1w) >= 21:
        # HMA formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_1w, half_n)
        wma_full = wma(close_1w, 21)
        # Align arrays (WMA reduces length by window-1)
        wma_half_aligned = np.concatenate([np.full(half_n - 1, np.nan), wma_half])
        wma_full_aligned = np.concatenate([np.full(20, np.nan), wma_full])
        raw_hma = 2 * wma_half_aligned - wma_full_aligned
        hma_21 = wma(raw_hma, sqrt_n)
        # Final alignment
        hma_21_aligned = np.concatenate([np.full(sqrt_n - 1, np.nan), hma_21])
        # Pad to match df_1w length
        if len(hma_21_aligned) < len(close_1w):
            hma_21_aligned = np.concatenate([hma_21_aligned, np.full(len(close_1w) - len(hma_21_aligned), np.nan)])
        elif len(hma_21_aligned) > len(close_1w):
            hma_21_aligned = hma_21_aligned[:len(close_1w)]
        
        # HMA is rising if current > previous, falling if current < previous
        hma_rising = np.roll(hma_21_aligned, 1) < hma_21_aligned
        hma_falling = np.roll(hma_21_aligned, 1) > hma_21_aligned
        # Handle first value
        hma_rising[0] = False
        hma_falling[0] = False
    else:
        hma_rising = np.full(len(df_1w), False)
        hma_falling = np.full(len(df_1w), False)
    
    # Align 1w HMA trend to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + HMA rising + volume filter
            if (donchian_breakout_up[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + HMA falling + volume filter
            elif (donchian_breakout_down[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint
            if donchian_revert_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint
            if donchian_revert_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals