#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w HMA21 trend filter and volume confirmation
# Long when price breaks above 1d Donchian high(20) AND 1w HMA21 rising (uptrend) AND volume > 1.3 * avg_volume(20) on 12h
# Short when price breaks below 1d Donchian low(20) AND 1w HMA21 falling (downtrend) AND volume > 1.3 * avg_volume(20) on 12h
# Exit when price crosses back through the 1d Donchian midpoint (high+low)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian(20) provides strong breakout levels that reduce whipsaw
# 1w HMA21 trend filter ensures we trade with the dominant weekly trend with less lag
# Volume confirmation (1.3x) validates breakout strength while limiting overtrading
# Works in both bull (breakouts with trend) and bear (breakdowns with trend) markets

name = "12h_1dDonchian20_1wHMA21_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) levels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_20 = high_20
    donchian_low_20 = low_20
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align 1d Donchian to 12h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_20)
    
    # Get 1w data ONCE before loop for HMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:  # Need at least 21 completed weekly bars for HMA21
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w HMA21 (Hull Moving Average)
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    n_half = 21 // 2
    n_sqrt = int(np.sqrt(21))
    
    # Calculate WMA for half period
    wma_half = np.full_like(close_1w, np.nan)
    for i in range(n_half - 1, len(close_1w)):
        wma_half[i] = wma(close_1w[i - n_half + 1:i + 1], n_half)
    
    # Calculate WMA for full period
    wma_full = np.full_like(close_1w, np.nan)
    for i in range(21 - 1, len(close_1w)):
        wma_full[i] = wma(close_1w[i - 21 + 1:i + 1], 21)
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt(n) period
    hma_21_1w = np.full_like(close_1w, np.nan)
    for i in range(n_sqrt - 1, len(raw_hma)):
        if not np.isnan(raw_hma[i - n_sqrt + 1:i + 1]).any():
            hma_21_1w[i] = wma(raw_hma[i - n_sqrt + 1:i + 1], n_sqrt)
    
    # Align 1w HMA21 to 12h timeframe
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian high, 1w HMA21 rising (uptrend), volume confirmation, in session
            if (close[i] > donchian_high_aligned[i] and 
                hma_21_1w_aligned[i] > hma_21_1w_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low, 1w HMA21 falling (downtrend), volume confirmation, in session
            elif (close[i] < donchian_low_aligned[i] and 
                  hma_21_1w_aligned[i] < hma_21_1w_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals