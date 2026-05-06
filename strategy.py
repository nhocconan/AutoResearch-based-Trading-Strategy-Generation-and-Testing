#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above upper Donchian channel AND 1d close > 1d EMA34 (uptrend) AND volume > 2.0 * 24-bar avg volume
# Short when price breaks below lower Donchian channel AND 1d close < 1d EMA34 (downtrend) AND volume > 2.0 * 24-bar avg volume
# Exit when price retraces to the 12h EMA20 (mean reversion to intermediate trend)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d EMA34 provides strong trend filter for better regime adaptation in both bull and bear markets
# Volume threshold set to 2.0x to reduce false breakouts while maintaining sufficient trade frequency

name = "12h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channel (20-bar) for 12h timeframe (based on previous bar)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    prev_high_20 = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    prev_low_20 = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 24-bar average volume (24*12h = 12d)
    avg_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * avg_volume_24)
    
    # Calculate 12h EMA20 for exit condition
    close_series = pd.Series(close)
    ema20_12h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(prev_high_20[i]) or np.isnan(prev_low_20[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(ema20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: Break above upper Donchian AND uptrend AND volume spike
            if close[i] > prev_high_20[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian AND downtrend AND volume spike
            elif close[i] < prev_low_20[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to 12h EMA20 (mean reversion)
            if close[i] <= ema20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to 12h EMA20 (mean reversion)
            if close[i] >= ema20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals