#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
# Long when: price breaks above Donchian upper band (20) AND 1w HMA(21) rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian lower band (20) AND 1w HMA(21) falling AND volume > 1.5x 20-period MA
# Exit when: price retouches Donchian middle band (10-period SMA of high/low) OR volume drops below average
# Uses Donchian for structure, 1w HMA for higher-timeframe trend filter, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian20_Breakout_1wHMA21_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 1d (20-period)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate volume confirmation on 1d using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for HMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Hull Moving Average (HMA) on 1w
    # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    if len(close_1w) >= 21:
        n = 21
        half_n = n // 2
        sqrt_n = int(np.sqrt(n))
        
        # WMA function
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # Calculate WMAs
        wma_full = np.full(len(close_1w), np.nan)
        wma_half = np.full(len(close_1w), np.nan)
        
        for i in range(n - 1, len(close_1w)):
            wma_full[i] = wma(close_1w[i - n + 1:i + 1], n)[-1]
        for i in range(half_n - 1, len(close_1w)):
            wma_half[i] = wma(close_1w[i - half_n + 1:i + 1], half_n)[-1]
        
        # HMA = WMA(2*WMA(half) - WMA(full), sqrt_n)
        hma_1w = np.full(len(close_1w), np.nan)
        for i in range(n - 1, len(close_1w)):
            if not np.isnan(wma_full[i]) and not np.isnan(wma_half[i]):
                diff = 2 * wma_half[i] - wma_full[i]
                hma_1w[i] = wma(close_1w[i - n + 1:i + 1], n)[-1] if i >= n - 1 else np.nan
                # Simplified: use the actual HMA formula correctly
                if i >= n - 1:
                    # Recalculate properly for HMA
                    wma_diff = np.full(len(close_1w), np.nan)
                    for j in range(n - 1, len(close_1w)):
                        wma_diff[j] = wma(close_1w[j - n + 1:j + 1], n)[-1] if j >= n - 1 else np.nan
                    # Actually, let's compute HMA step by step correctly
                    wma_half_vals = np.full(len(close_1w), np.nan)
                    wma_full_vals = np.full(len(close_1w), np.nan)
                    for j in range(half_n - 1, len(close_1w)):
                        wma_half_vals[j] = wma(close_1w[j - half_n + 1:j + 1], half_n)[-1]
                    for j in range(n - 1, len(close_1w)):
                        wma_full_vals[j] = wma(close_1w[j - n + 1:j + 1], n)[-1]
                    # Now HMA
                    if i >= n - 1:
                        hma_input = 2 * wma_half_vals[i] - wma_full_vals[i]
                        # We need WMA of hma_input over sqrt_n period
                        # But since we don't have history of hma_input, approximate:
                        hma_1w[i] = (2 * wma_half_vals[i] - wma_full_vals[i])  # Simplified proxy
        
        # Much simpler approach: use HMA approximation via WMA
        # HMA(21) ≈ WMA(2*WMA(10.5) - WMA(21), 4) since sqrt(21)≈4.58→4
        half_n = 10
        wma_21 = np.full(len(close_1w), np.nan)
        wma_10 = np.full(len(close_1w), np.nan)
        for i in range(20, len(close_1w)):
            wma_21[i] = wma(close_1w[i-20:i+1], 21)[-1]
        for i in range(9, len(close_1w)):
            wma_10[i] = wma(close_1w[i-9:i+1], 10)[-1]
        
        hma_1w = np.full(len(close_1w), np.nan)
        for i in range(20, len(close_1w)):
            if not np.isnan(wma_21[i]) and not np.isnan(wma_10[i]):
                hma_1w[i] = 2 * wma_10[i] - wma_21[i]  # Simplified HMA
                # Apply WMA of this over 4 periods
                if i >= 23:  # 20 + 3 for 4-period WMA
                    hma_4 = np.full(len(close_1w), np.nan)
                    for j in range(3, len(close_1w)):
                        hma_4[j] = wma(hma_1w[j-3:j+1], 4)[-1] if j >= 3 else np.nan
                    hma_1w = hma_4
        
        # Even simpler: just use the smoothed median
        # Let's use a proper HMA implementation
        def calculate_hma(values, window):
            half = window // 2
            sqrt = int(np.sqrt(window))
            
            wma_half = np.full(len(values), np.nan)
            wma_full = np.full(len(values), np.nan)
            
            for i in range(half - 1, len(values)):
                wma_half[i] = np.nansum(values[i - half + 1:i + 1] * np.arange(1, half + 1)) / (half * (half + 1) / 2)
            for i in range(window - 1, len(values)):
                wma_full[i] = np.nansum(values[i - window + 1:i + 1] * np.arange(1, window + 1)) / (window * (window + 1) / 2)
            
            hma = np.full(len(values), np.nan)
            for i in range(window - 1, len(values)):
                if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
                    diff = 2 * wma_half[i] - wma_full[i]
                    if i >= half - 1:  # enough for WMA of diff
                        wma_diff = np.nansum(diff * np.ones(half)) / half if half > 0 else diff  # Simplified
                        hma[i] = wma_diff  # Further simplified
            
            # Apply WMA of the diff over sqrt period
            hma_final = np.full(len(values), np.nan)
            for i in range(window - 1, len(values)):
                if not np.isnan(hma[i]):
                    hma_final[i] = hma[i]  # Placeholder - using simplified HMA
            
            return hma_final
        
        # Use a working HMA approximation
        def wma_simple(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            result = np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            for i in range(window - 1, len(values)):
                result[i] = np.sum(values[i - window + 1:i + 1] * weights) / np.sum(weights)
            return result
        
        wma_21 = wma_simple(close_1w, 21)
        wma_10 = wma_simple(close_1w, 10)
        hma_raw = 2 * wma_10 - wma_21
        hma_1w = wma_simple(hma_raw, 4)  # sqrt(21)≈4.58→4
    else:
        hma_1w = np.full(len(close_1w), np.nan)
    
    # Align 1w HMA to 1d timeframe
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate HMA slope for trend (rising/falling)
    hma_1w_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]):
            hma_1w_slope[i] = hma_1w_aligned[i] - hma_1w_aligned[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(hma_1w_slope[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian upper AND HMA rising AND volume filter
            if (close[i] > donchian_high[i] and 
                hma_1w_slope[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian lower AND HMA falling AND volume filter
            elif (close[i] < donchian_low[i] and 
                  hma_1w_slope[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian mid OR HMA slope turns negative OR volume drops
            if (close[i] < donchian_mid[i] or 
                hma_1w_slope[i] <= 0 or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian mid OR HMA slope turns positive OR volume drops
            if (close[i] > donchian_mid[i] or 
                hma_1w_slope[i] >= 0 or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals