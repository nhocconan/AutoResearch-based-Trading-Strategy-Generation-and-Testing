#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w HMA21 trend filter
# Long when: price breaks above upper Donchian(20), volume > 2.0x 48-period average (1w equivalent), and close > 1w HMA21
# Short when: price breaks below lower Donchian(20), volume > 2.0x 48-period average, and close < 1w HMA21
# Exit when price returns to the opposite Donchian band (mean reversion)
# Uses Donchian structure for clear breakouts, volume confirmation on 1w for conviction, 1w HMA for major trend filter
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian20_Breakout_1wHMA21_VolumeSpike"
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
    open_price = prices['open'].values
    
    # Calculate Donchian(20) bands from previous 20 periods (lookback)
    if len(high) >= 20:
        # Use rolling window of 20 on historical data (excluding current bar)
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
        donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Get 1w data ONCE before loop for volume and HMA filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 48:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w volume confirmation using 48-period MA (equivalent to 1d lookback)
    if len(volume_1w) >= 48:
        vol_ma_48 = pd.Series(volume_1w).rolling(window=48, min_periods=48).mean().values
        volume_filter_1w = volume_1w > (2.0 * vol_ma_48)
    else:
        volume_filter_1w = np.zeros(len(volume_1w), dtype=bool)
    
    # Calculate 1w HMA21 trend filter
    if len(close_1w) >= 21:
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = np.array([wma(close_1w[i:i+half_len], half_len) if i+half_len <= len(close_1w) else np.nan 
                            for i in range(len(close_1w))])
        wma_full = np.array([wma(close_1w[i:i+21], 21) if i+21 <= len(close_1w) else np.nan 
                            for i in range(len(close_1w))])
        raw_hma = 2 * wma_half - wma_full
        hma_21 = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(raw_hma) else np.nan 
                          for i in range(len(raw_hma))])
    else:
        hma_21 = np.full(len(close_1w), np.nan)
    
    # Align 1w indicators to 1d timeframe
    volume_filter_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_filter_1w.astype(float))
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(volume_filter_1w_aligned[i]) or 
            np.isnan(hma_21_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, volume filter, and above 1w HMA21
            if (close[i] > donchian_high[i] and 
                open_price[i] <= donchian_high[i] and  # Ensure breakout happens on this bar
                volume_filter_1w_aligned[i] > 0.5 and  # Boolean as float
                close[i] > hma_21_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian, volume filter, and below 1w HMA21
            elif (close[i] < donchian_low[i] and 
                  open_price[i] >= donchian_low[i] and  # Ensure breakdown happens on this bar
                  volume_filter_1w_aligned[i] > 0.5 and  # Boolean as float
                  close[i] < hma_21_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below lower Donchian band (mean reversion to opposite band)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above upper Donchian band (mean reversion to opposite band)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals