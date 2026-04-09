#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA(21) trend filter and volume confirmation
# Uses Donchian channels from 4h data: breakout above upper band = long, below lower band = short
# 1d HMA21 filter ensures trades align with higher timeframe trend
# Volume confirmation (current > 1.5x 20-bar average) reduces false breakouts
# Designed for 4h timeframe to target 20-50 trades/year (75-200 over 4 years)
# Works in bull/bear: HMA21 adapts to trend, Donchian provides robust structure

name = "4h_1d_donchian_hma_volume_v3"
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
    
    # Load daily data ONCE before loop for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) from 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_20 = high_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d HMA21 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    half_n = int(21/2 + 0.5)
    wma_half = close_1d_series.rolling(window=half_n, min_periods=half_n).mean()
    wma_full = close_1d_series.rolling(window=21, min_periods=21).mean()
    hma_21_1d = (2 * wma_half - wma_full).values
    
    # Align 1d HMA21 to 4h timeframe
    hma_21_4h = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(hma_21_4h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR trend turns bearish
            if close[i] < lower_20[i] or close[i] < hma_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR trend turns bullish
            if close[i] > upper_20[i] or close[i] > hma_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Donchian upper band AND price > 1d HMA21 (bullish trend)
                if close[i] > upper_20[i] and close[i] > hma_21_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian lower band AND price < 1d HMA21 (bearish trend)
                elif close[i] < lower_20[i] and close[i] < hma_21_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals