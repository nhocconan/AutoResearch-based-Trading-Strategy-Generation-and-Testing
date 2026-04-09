#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation (1.5x 20-bar avg)
# Donchian breakouts capture strong momentum moves; 1d HMA21 ensures alignment with daily trend
# Volume confirmation filters weak breakouts; discrete sizing 0.25 minimizes fee drag
# Works in bull/bear: HMA trend filter avoids counter-trend trades in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on BTC/ETH/SOL

name = "12h_1d_donchian_hma_volume_v1"
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
    
    # Load 1d data ONCE before loop for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA(21) for trend direction
    close_1d = df_1d['close'].values
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Load 1d data ONCE before loop for Donchian(20) channels
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels: 20-period high/low
    donchian_high_20 = np.full(len(df_1d), np.nan)
    donchian_low_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        donchian_high_20[i] = np.max(high_1d[i-20:i])
        donchian_low_20[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian channels to 12h timeframe (completed 1d bar only)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low(20) OR price < 1d HMA21 (trend change)
            if close[i] < donchian_low_20_aligned[i] or close[i] < hma_21_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high(20) OR price > 1d HMA21 (trend change)
            if close[i] > donchian_high_20_aligned[i] or close[i] > hma_21_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + HMA21 trend filter
            if volume_confirmed:
                # Long entry: price > Donchian high(20) AND price > 1d HMA21 (bullish breakout + uptrend)
                if close[i] > donchian_high_20_aligned[i] and close[i] > hma_21_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low(20) AND price < 1d HMA21 (bearish breakout + downtrend)
                elif close[i] < donchian_low_20_aligned[i] and close[i] < hma_21_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals

def calculate_hma(values, period):
    """Calculate Hull Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.full_like(values, np.nan)
    for i in range(half_period, len(values)):
        wma_half[i] = np.average(values[i-half_period:i], weights=np.arange(1, half_period+1))
    
    # WMA of full period
    wma_full = np.full_like(values, np.nan)
    for i in range(period, len(values)):
        wma_full[i] = np.average(values[i-period:i], weights=np.arange(1, period+1))
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = np.full_like(values, np.nan)
    for i in range(sqrt_period, len(values)):
        hma[i] = np.average(raw_hma[i-sqrt_period:i], weights=np.arange(1, sqrt_period+1))
    
    return hma