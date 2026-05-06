#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# Long when price breaks above Donchian(20) high AND close > 1d EMA50 (uptrend) AND volume > 1.5 * 20-bar avg volume
# Short when price breaks below Donchian(20) low AND close < 1d EMA50 (downtrend) AND volume > 1.5 * 20-bar avg volume
# Exit when price crosses opposite Donchian(10) level or trend reverses (close crosses 1d EMA50)
# Uses discrete sizing 0.25 to control fee drag and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian channels provide clear structure; 1d EMA50 ensures higher-timeframe trend alignment
# Volume spike confirms institutional participation; Donchian(10) exit allows profits to run

name = "4h_Donchian20_1dEMA50_Volume_v1"
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
    
    # Calculate Donchian(20) for entry
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian(10) for exit
    donchian_high_10 = high_series.rolling(window=10, min_periods=10).max().values
    donchian_low_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: price > Donchian(20) high AND uptrend AND volume spike
            if close[i] > donchian_high_20[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian(20) low AND downtrend AND volume spike
            elif close[i] < donchian_low_20[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian(10) low OR trend reverses (close < 1d EMA50)
            if close[i] < donchian_low_10[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian(10) high OR trend reverses (close > 1d EMA50)
            if close[i] > donchian_high_10[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals