#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND 1d close > 1d EMA50 AND volume > 1.5 * 20-bar avg volume
# Short when price breaks below Donchian lower band AND 1d close < 1d EMA50 AND volume > 1.5 * 20-bar avg volume
# Exit when price retests the Donchian midpoint (mean of upper and lower bands)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Donchian provides robust structure, EMA50 filters higher timeframe trend, volume reduces false breakouts

name = "12h_Donchian20_1dEMA50_Volume_v1"
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
    
    # Calculate 1d Donchian(20) levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian bands from previous 20 completed 1d bars (avoid look-ahead)
    donchian_upper = np.zeros(len(high_1d))
    donchian_lower = np.zeros(len(high_1d))
    donchian_mid = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        if i < 20:
            # For first 20 bars, use available history (will be aligned properly)
            start_idx = 0
            end_idx = i + 1
        else:
            start_idx = i - 19
            end_idx = i + 1
        
        if end_idx > start_idx:
            donchian_upper[i] = np.max(high_1d[start_idx:end_idx])
            donchian_lower[i] = np.min(low_1d[start_idx:end_idx])
            donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2
        else:
            donchian_upper[i] = high_1d[i]
            donchian_lower[i] = low_1d[i]
            donchian_mid[i] = close_1d[i]
    
    # Align Donchian levels to 12h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper band AND uptrend AND volume confirmation
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower band AND downtrend AND volume confirmation
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests midpoint from above
            if close[i] <= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests midpoint from below
            if close[i] >= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals