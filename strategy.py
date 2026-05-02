#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + HMA(21) trend filter + volume confirmation
# Donchian breakout captures momentum, HMA confirms trend direction, volume ensures conviction
# Works in bull/bear by requiring both price channel breakout and trend alignment
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Discrete sizing 0.30 balances profit potential and fee drag

name = "4h_Donchian20_HMA21_Volume_Trend"
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
    
    # Calculate 1d HMA(21) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def calculate_wma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    def calculate_hma(values, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = calculate_wma(values, half_period)
        wma_full = calculate_wma(values, period)
        if len(wma_half) == 0 or len(wma_full) == 0:
            return np.full_like(values, np.nan)
        # Align arrays: wma_half starts at index half_period-1, wma_full at index period-1
        # We need to align them to the same starting point
        raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
        hma = calculate_wma(raw_hma, sqrt_period)
        # Pad with NaN to match original length
        result = np.full_like(values, np.nan)
        result[period-1:period-1+len(hma)] = hma
        return result
    
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate Donchian channels (20-period)
    def calculate_donchian(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, 20)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(donchian_20_upper[i]) or 
            np.isnan(donchian_20_lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper AND price > 1d HMA21 (uptrend) AND volume spike
            if (close[i] > donchian_20_upper[i] and 
                close[i] > hma_21_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: Price breaks below Donchian lower AND price < 1d HMA21 (downtrend) AND volume spike
            elif (close[i] < donchian_20_lower[i] and 
                  close[i] < hma_21_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower OR price below 1d HMA21 (trend change)
            if close[i] < donchian_20_lower[i] or close[i] < hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper OR price above 1d HMA21 (trend change)
            if close[i] > donchian_20_upper[i] or close[i] > hma_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals