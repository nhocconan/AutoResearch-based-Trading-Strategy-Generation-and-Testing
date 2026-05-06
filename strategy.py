#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 12h HMA21 is rising AND volume > 1.5 * avg_volume(20)
# Short when price breaks below Donchian(20) low AND 12h HMA21 is falling AND volume > 1.5 * avg_volume(20)
# Exit when price crosses Donchian(20) midpoint (mean reversion)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian breakouts capture strong momentum moves
# 12h HMA21 trend filter ensures we trade with the dominant intermediate trend
# Volume confirmation validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "4h_Donchian20_Breakout_12hHMA21_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:  # Need at least 21 completed 12h bars for HMA21
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h HMA21: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    wma_half = np.array([wma(close_12h[i:i+half_len], half_len) if i+half_len <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    wma_full = np.array([wma(close_12h[i:i+21], 21) if i+21 <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(raw_hma) else np.nan 
                           for i in range(len(raw_hma))])
    
    # Align 12h HMA21 to 4h timeframe (wait for completed 12h bar)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate Donchian(20) channels on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian(20) high AND HMA21 rising AND volume spike
            if (close[i] > highest_high_20[i] and 
                hma_21_12h_aligned[i] > hma_21_12h_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low AND HMA21 falling AND volume spike
            elif (close[i] < lowest_low_20[i] and 
                  hma_21_12h_aligned[i] < hma_21_12h_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals