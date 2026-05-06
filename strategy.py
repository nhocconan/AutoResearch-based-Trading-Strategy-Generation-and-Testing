#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter
# Long when price breaks above Donchian upper (20-bar high) AND volume > 1.5 * 20-bar avg volume AND close > 1d EMA50
# Short when price breaks below Donchian lower (20-bar low) AND volume > 1.5 * 20-bar avg volume AND close < 1d EMA50
# Exit when price crosses opposite Donchian level (lower for long exit, upper for short exit)
# Uses discrete sizing 0.28 to balance return and drawdown
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# Donchian channels provide clear structural breakouts; volume confirms institutional participation
# 1d EMA50 ensures alignment with higher-timeframe trend to reduce whipsaw in ranging markets

name = "4h_Donchian20_1dEMA50_VolumeBreakout_v1"
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
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: price breaks above upper band AND uptrend AND volume spike
            if close[i] > donchian_upper[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.28
                position = 1
            # Short: price breaks below lower band AND downtrend AND volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian lower band (structure break)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: price crosses above Donchian upper band (structure break)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals