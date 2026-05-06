#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d HMA21 trend filter and volume spike confirmation
# Long when price breaks above R4 AND close > 1d HMA21 (uptrend) AND volume > 2.5 * 20-bar avg volume
# Short when price breaks below S4 AND close < 1d HMA21 (downtrend) AND volume > 2.5 * 20-bar avg volume
# Exit when price retraces to the Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d HMA21 provides adaptive trend filter that reduces whipsaw in ranging markets
# Volume spike threshold increased to 2.5x to significantly reduce false breakouts and lower trade frequency
# Pivot exit works in ranging markets and captures mean reversion after breakout failure
# Strategy avoids overtrading by requiring strong confluence of breakout, trend, and volume

name = "4h_Camarilla_R4S4_1dHMA21_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for 4h timeframe (based on previous bar)
    # Camarilla: Pivot = (H + L + C) / 3
    # R4 = Pivot + (H - L) * 1.1
    # S4 = Pivot - (H - L) * 1.1
    pivot = (high + low + close) / 3.0
    r4 = pivot + (high - low) * 1.1
    s4 = pivot - (high - low) * 1.1
    
    # Shift by 1 to use only completed bar data (no look-ahead)
    r4_prev = np.roll(r4, 1)
    s4_prev = np.roll(s4, 1)
    pivot_prev = np.roll(pivot, 1)
    r4_prev[0] = np.nan
    s4_prev[0] = np.nan
    pivot_prev[0] = np.nan
    
    # Get 1d data ONCE before loop for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA21 (Hull Moving Average)
    # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Pad arrays for WMA calculation
    wma_full = lambda arr, window: np.convolve(arr, np.arange(1, window+1), mode='valid') / (window*(window+1)/2)
    
    # Calculate WMA for full array
    wma_21 = np.full_like(close_1d, np.nan)
    wma_half = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 20:  # 21-period WMA needs 21 values
            wma_21[i] = np.dot(close_1d[i-20:i+1], np.arange(1, 22)) / (21*22/2)
        if i >= half_n-1:  # half-period WMA
            wma_half[i] = np.dot(close_1d[i-half_n+1:i+1], np.arange(1, half_n+1)) / (half_n*(half_n+1)/2)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_raw = 2 * wma_half - wma_21
    hma_21 = np.full_like(close_1d, np.nan)
    for i in range(len(hma_raw)):
        if i >= sqrt_n-1 and not np.isnan(hma_raw[i-sqrt_n+1:i+1]).any():
            hma_21[i] = np.dot(hma_raw[i-sqrt_n+1:i+1], np.arange(1, sqrt_n+1)) / (sqrt_n*(sqrt_n+1)/2)
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # Calculate volume confirmation: volume > 2.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r4_prev[i]) or np.isnan(s4_prev[i]) or 
            np.isnan(pivot_prev[i]) or np.isnan(hma_21_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend and volume filters
            # Long: Break above R4 AND uptrend AND volume spike
            if close[i] > r4_prev[i] and close[i] > hma_21_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S4 AND downtrend AND volume spike
            elif close[i] < s4_prev[i] and close[i] < hma_21_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to pivot point (mean reversion)
            if close[i] <= pivot_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to pivot point (mean reversion)
            if close[i] >= pivot_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals