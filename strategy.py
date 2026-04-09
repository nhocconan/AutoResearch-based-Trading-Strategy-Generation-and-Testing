#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (HMA50) and volume confirmation
# Uses Donchian channels from 1d data: breakout above upper band = long, below lower band = short
# 1w HMA50 filter ensures trades align with higher timeframe trend (more stable than 1d)
# Volume confirmation reduces false breakouts
# Designed for 1d timeframe to target 10-30 trades/year (30-100 over 4 years)
# Works in bull/bear: HMA50 adapts to trend, Donchian provides robust structure

name = "1d_1w_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from 1d data
    high_1d = high
    low_1d = low
    
    # Upper band: highest high of last 20 periods
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w HMA50 trend filter
    close_1w = df_1w['close'].values
    half_n = int(50/2 + 0.5)
    wma_half = pd.Series(close_1w).rolling(window=half_n, min_periods=half_n).mean()
    wma_full = pd.Series(close_1w).rolling(window=50, min_periods=50).mean()
    hma_50_1w = (2 * wma_half - wma_full).values
    
    # Align 1w HMA50 to 1d timeframe
    hma_50_1d = align_htf_to_ltf(prices, df_1w, hma_50_1w)
    
    # Calculate 20-period average volume for volume confirmation (1d volume)
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
            np.isnan(hma_50_1d[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR trend turns bearish
            if close[i] < lower_20[i] or close[i] < hma_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR trend turns bullish
            if close[i] > upper_20[i] or close[i] > hma_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Donchian upper band AND price > 1w HMA50 (bullish trend)
                if close[i] > upper_20[i] and close[i] > hma_50_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian lower band AND price < 1w HMA50 (bearish trend)
                elif close[i] < lower_20[i] and close[i] < hma_50_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals