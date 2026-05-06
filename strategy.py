#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above upper Donchian AND close > 12h EMA50 (uptrend) AND volume > 1.5 * 20-bar avg volume
# Short when price breaks below lower Donchian AND close < 12h EMA50 (downtrend) AND volume > 1.5 * 20-bar avg volume
# Exit when price retouches the middle of the Donchian channel (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to control fee drag and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian provides clear structure; 12h EMA50 ensures higher-timeframe trend alignment
# Volume spike confirms institutional participation; middle band exit works in ranging markets
# Specifically designed for 4h timeframe to balance trade frequency and capture significant moves

name = "4h_Donchian20_12hEMA50_VolumeBreakout_v1"
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
    
    # Calculate Donchian channels for 4h timeframe (20-period)
    # Upper band: highest high of last 20 bars
    # Lower band: lowest low of last 20 bars
    # Middle band: average of upper and lower
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    middle = (upper + lower) / 2.0
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: Break above upper band AND uptrend AND volume spike
            if close[i] > upper[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND downtrend AND volume spike
            elif close[i] < lower[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retouches middle band (mean reversion)
            if close[i] <= middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retouches middle band (mean reversion)
            if close[i] >= middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals