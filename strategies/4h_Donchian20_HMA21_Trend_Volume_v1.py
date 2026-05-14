#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation
# Uses 4h timeframe for signal generation with Donchian channel breakouts
# 12h HMA(21) determines primary trend direction (bullish/bearish) - multi-timeframe alignment
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) balances return and risk while minimizing fee drag
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Donchian provides objective price channels based on recent highs/lows
# HMA trend filter ensures trades only in higher timeframe trend direction
# Works in both bull and bear markets by only taking trades aligned with 12h trend

name = "4h_Donchian20_HMA21_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h HMA(21) for trend determination
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # Calculate WMA for 12h data
    wma_half = pd.Series(close_12h).rolling(window=half_len, min_periods=half_len).apply(
        lambda x: np.dot(x, np.arange(1, half_len+1)) / np.arange(1, half_len+1).sum(), raw=True
    ).values
    wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).apply(
        lambda x: np.dot(x, np.arange(1, 22)) / np.arange(1, 22).sum(), raw=True
    ).values
    
    # HMA = WMA(2*WMA(half) - WMA(full)), sqrt_len
    hma_12h = 2 * wma_half - wma_full
    hma_12h = pd.Series(hma_12h).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
        lambda x: np.dot(x, np.arange(1, sqrt_len+1)) / np.arange(1, sqrt_len+1).sum(), raw=True
    ).values
    
    # Align 12h HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Donchian Upper + volume confirm + close > 12h HMA (bullish trend)
            if close[i] > donchian_high[i] and volume_confirm[i] and close[i] > hma_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian Lower + volume confirm + close < 12h HMA (bearish trend)
            elif close[i] < donchian_low[i] and volume_confirm[i] and close[i] < hma_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Donchian Lower or close < 12h HMA (trend reversal)
            if close[i] < donchian_low[i] or close[i] < hma_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Donchian Upper or close > 12h HMA (trend reversal)
            if close[i] > donchian_high[i] or close[i] > hma_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals