# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-week Donchian channel breakout with 1-day volume confirmation and 1-day ADX trend filter.
# In bull markets: Buy when price breaks above weekly Donchian high (20-period) with volume > 1.5x average and ADX > 25.
# In bear markets: Sell when price breaks below weekly Donchian low (20-period) with volume > 1.5x average and ADX > 25.
# Weekly structure captures major trend, daily volume confirms institutional participation, ADX filters choppy markets.
# Target: 12-37 trades per year (50-150 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channel and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = np.full(len(high_1w), np.nan)
    donchian_low = np.full(len(low_1w), np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Calculate weekly ADX (14-period) for trend strength
    # ADX requires +DI, -DI, and TR
    tr = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - high_1w[:-1]),
            np.abs(low_1w[1:] - low_1w[:-1])
        )
    )
    # Prepend first TR as 0 for alignment
    tr = np.concatenate([[0], tr])
    
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    # Prepend first values as 0
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth with Wilder's smoothing (EMA-like with alpha=1/period)
    def wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilder_smoothing(tr, 14)
    plus_di_smoothed = wilder_smoothing(plus_dm, 14)
    minus_di_smoothed = wilder_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    dx = np.where((plus_di_smoothed + minus_di_smoothed) != 0,
                  100 * np.abs(plus_di_smoothed - minus_di_smoothed) / (plus_di_smoothed + minus_di_smoothed),
                  0)
    adx = wilder_smoothing(dx, 14)
    
    # Calculate daily average volume (20-period)
    avg_volume = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align all indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_aligned[i]
        adx_val = adx_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume confirmation + strong trend
            if (price > upper and
                volume_confirm and
                strong_trend):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian low + volume confirmation + strong trend
            elif (price < lower and
                  volume_confirm and
                  strong_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below Donchian low or trend weakens
            if (price < lower or
                adx_val < 20):  # Exit when trend weakens
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above Donchian high or trend weakens
            if (price > upper or
                adx_val < 20):  # Exit when trend weakens
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Donchian_ADX_Volume"
timeframe = "12h"
leverage = 1.0