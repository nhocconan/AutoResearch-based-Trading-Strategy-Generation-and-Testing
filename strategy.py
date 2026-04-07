#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend and Volume Filter
Long when price breaks above 20-day Donchian high with weekly uptrend and volume confirmation
Short when price breaks below 20-day Donchian low with weekly downtrend and volume confirmation
Exit when price reverses to midpoint of Donchian channel
Designed for low-frequency, high-conviction trades to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20-day) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === Weekly Trend (HMA 21) ===
    df_1w = get_htf_data(prices, '1w')
    hma_21 = calculate_hma(df_1w['close'].values, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # === Volume Confirmation (20-day average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint (mean reversion)
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint (mean reversion)
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with weekly trend alignment
            if close[i] > donchian_high[i] and hma_21_aligned[i] > hma_21_aligned[i-1]:
                # Break above upper band with rising weekly trend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low[i] and hma_21_aligned[i] < hma_21_aligned[i-1]:
                # Break below lower band with falling weekly trend -> short
                position = -1
                signals[i] = -0.25
    
    return signals

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA function
    def wma(s, n):
        if len(s) < n:
            return np.full_like(s, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(s, weights, mode='valid') / (weights.sum() * np.ones_like(s[:len(s)-n+1]) if len(s) >= n else np.array([]))
    
    # Handle edge cases for convolution
    wma_half = np.full_like(series, np.nan)
    wma_full = np.full_like(series, np.nan)
    
    if len(series) >= half_period:
        weights_half = np.arange(1, half_period + 1)
        conv_half = np.convolve(series, weights_half, mode='valid')
        wma_half[half_period-1:] = conv_half / weights_half.sum()
    
    if len(series) >= period:
        weights_full = np.arange(1, period + 1)
        conv_full = np.convolve(series, weights_full, mode='valid')
        wma_full[period-1:] = conv_full / weights_full.sum()
    
    # Calculate HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of sqrt period
    if len(raw_hma) >= sqrt_period:
        # Remove leading NaNs for convolution
        valid_start = int(np.sqrt(period)) - 1 if not np.isnan(raw_hma).all() else 0
        valid_raw = raw_hma[~np.isnan(raw_hma)]
        if len(valid_raw) >= sqrt_period:
            weights_sqrt = np.arange(1, sqrt_period + 1)
            conv_sqrt = np.convolve(valid_raw, weights_sqrt, mode='valid')
            wma_sqrt = conv_sqrt / weights_sqrt.sum()
            # Reconstruct full array with NaNs
            hma_result = np.full_like(raw_hma, np.nan)
            # Find where to place the valid WMA values
            first_valid_idx = np.where(~np.isnan(raw_hma))[0][0] if np.any(~np.isnan(raw_hma)) else 0
            hma_result[first_valid_idx + sqrt_period - 1:first_valid_idx + sqrt_period - 1 + len(wma_sqrt)] = wma_sqrt
            return hma_result
    
    return np.full_like(series, np.nan)