#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA(21) trend filter and volume confirmation
# Uses 4h timeframe for signal generation with Donchian channel breakouts
# 1d HMA(21) determines primary trend direction - multi-timeframe alignment
# Volume spike (1.8x 20-period average) ensures strong institutional participation
# Discrete position sizing (0.25) minimizes fee drag while maintaining profitability
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe
# Donchian channels provide adaptive support/resistance based on recent price action
# Works in both bull and bear markets by only taking trades aligned with 1d HMA trend
# Prioritizes BTC/ETH over SOL by requiring volume confirmation and trend alignment

name = "4h_Donchian20_1dHMA21_Trend_Volume_v1"
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
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d HMA(21) for trend determination
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    wma_half = np.array([np.nan] * len(close_1d))
    wma_full = np.array([np.nan] * len(close_1d))
    
    for i in range(half_len, len(close_1d)):
        wma_half[i] = wma(close_1d[i-half_len+1:i+1], half_len)
    
    for i in range(21, len(close_1d)):
        wma_full[i] = wma(close_1d[i-21+1:i+1], 21)
    
    raw_hma = 2 * wma_half - wma_full
    hma_21_1d = np.array([np.nan] * len(close_1d))
    
    for i in range(sqrt_len, len(raw_hma)):
        if not np.isnan(raw_hma[i]):
            hma_21_1d[i] = wma(raw_hma[i-sqrt_len+1:i+1], sqrt_len)
    
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate Donchian(20) channels on 4h
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Upper Donchian + volume spike + close > 1d HMA21 (bullish trend)
            if close[i] > upper_channel[i] and volume_spike[i] and close[i] > hma_21_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Lower Donchian + volume spike + close < 1d HMA21 (bearish trend)
            elif close[i] < lower_channel[i] and volume_spike[i] and close[i] < hma_21_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Lower Donchian or close < 1d HMA21 (trend reversal)
            if close[i] < lower_channel[i] or close[i] < hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Upper Donchian or close > 1d HMA21 (trend reversal)
            if close[i] > upper_channel[i] or close[i] > hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals