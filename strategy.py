#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Donchian(20) breakout with 1d volume confirmation and ATR filter
# Long when price breaks above 1w Donchian high(20) AND 1d volume > 1.5 * avg_volume(20) AND 1d ATR ratio > 0.8
# Short when price breaks below 1w Donchian low(20) AND 1d volume > 1.5 * avg_volume(20) AND 1d ATR ratio > 0.8
# Exit when price crosses 1w Donchian midline (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 1w Donchian provides weekly structure with proven breakout edge
# Volume confirmation reduces false breakouts (institutional participation)
# ATR filter ensures sufficient volatility for meaningful moves

name = "6h_1wDonchian20_1dVolumeSpike_ATR_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2.0
        return upper, lower, middle
    
    donchian_upper_1w, donchian_lower_1w, donchian_middle_1w = donchian_channels(high_1w, low_1w, period=20)
    
    # Get 1d data ONCE before loop for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ATR and volume average
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    
    # Calculate 1d ATR for volatility filter
    def calculate_atr(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0  # First period has no prior close
        
        # Wilder's ATR
        atr = np.full_like(close, np.nan, dtype=float)
        if len(close) < period:
            return atr
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, period=14)
    # ATR ratio: current ATR / 20-period average ATR (to filter low volatility periods)
    avg_atr_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / avg_atr_20_1d
    atr_filter_1d = atr_ratio_1d > 0.8  # Sufficient volatility for meaningful moves
    
    # Align 1w Donchian levels to 6h timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle_1w)
    
    # Align 1d indicators to 6h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(atr_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper with volume spike and sufficient volatility
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_spike_aligned[i] and atr_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower with volume spike and sufficient volatility
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_spike_aligned[i] and atr_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w Donchian middle (mean reversion)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w Donchian middle (mean reversion)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals