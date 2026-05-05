#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d HMA21 Trend Filter + Volume Spike
# Long when price breaks above 20-period Donchian high (1d) AND price > 1d HMA21 (uptrend) AND volume spike
# Short when price breaks below 20-period Donchian low (1d) AND price < 1d HMA21 (downtrend) AND volume spike
# Donchian channels provide clear structure with fewer, higher-quality breaks than Camarilla
# HMA21 offers smoother trend with less lag than EMA for better trend identification
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced for 12h timeframe)
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag on 12h
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)

name = "12h_Donchian20_1dHMA21_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian and HMA21
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d HMA21 (Hull Moving Average)
    close_1d = df_1d['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function for HMA calculation
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Calculate HMA: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = np.array([np.nan] * len(close_1d))
    wma_full = np.array([np.nan] * len(close_1d))
    
    for i in range(half_len, len(close_1d)):
        wma_half[i] = wma(close_1d[i-half_len+1:i+1], half_len)
    
    for i in range(21, len(close_1d)):
        wma_full[i] = wma(close_1d[i-21+1:i+1], 21)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of the diff with period sqrt(n)
    hma_21_1d = np.array([np.nan] * len(close_1d))
    for i in range(sqrt_len, len(diff)):
        if not np.isnan(diff[i-sqrt_len+1:i+1]).any():
            hma_21_1d[i] = wma(diff[i-sqrt_len+1:i+1], sqrt_len)
    
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate Donchian channels (20-period) from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: highest high over last 20 periods
    donchian_high = np.full(len(high_1d), np.nan)
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20+1:i+1])
    
    # Donchian low: lowest low over last 20 periods
    donchian_low = np.full(len(low_1d), np.nan)
    for i in range(20, len(low_1d)):
        donchian_low[i] = np.min(low_1d[i-20+1:i+1])
    
    # Shift by 1 to use only completed daily bar
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_shifted)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_shifted)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to calculations or insufficient data)
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND uptrend (price > HMA21) AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > hma_21_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND downtrend (price < HMA21) AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < hma_21_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian high OR closes below HMA21
            if close[i] < donchian_high_aligned[i] or close[i] < hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian low OR closes above HMA21
            if close[i] > donchian_low_aligned[i] or close[i] > hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals