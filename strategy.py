#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter (HMA50) and volume confirmation
# Uses Camarilla levels from 1d data: breakout above H3 = long, below L3 = short
# 1d HMA50 filter ensures trades align with higher timeframe trend
# Volume confirmation reduces false breakouts
# Designed for 12h timeframe to target 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: HMA50 adapts to trend, Camarilla provides robust structure

name = "12h_1d_camarilla_hma_volume_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels and HMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Using rolling window of 1 day (previous bar) for OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    range_ = prev_high - prev_low
    h3 = prev_close + range_ * 1.1 / 4
    l3 = prev_close - range_ * 1.1 / 4
    h4 = prev_close + range_ * 1.1 / 2
    l4 = prev_close - range_ * 1.1 / 2
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3)
    h4_12h = align_htf_to_ltf(prices, df_1d, h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate 1d HMA50 trend filter
    half_n = int(50/2 + 0.5)
    wma_half = pd.Series(df_1d['close']).rolling(window=half_n, min_periods=half_n).mean()
    wma_full = pd.Series(df_1d['close']).rolling(window=50, min_periods=50).mean()
    hma_50_1d = (2 * wma_half - wma_full).values
    
    # Align 1d HMA50 to 12h timeframe
    hma_50_12h = align_htf_to_ltf(prices, df_1d, hma_50_1d)
    
    # Calculate 20-period average volume for volume confirmation (12h volume)
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
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or
            np.isnan(hma_50_12h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR trend turns bearish
            if close[i] < l3_12h[i] or close[i] < hma_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR trend turns bullish
            if close[i] > h3_12h[i] or close[i] > hma_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Camarilla H3 AND price > 1d HMA50 (bullish trend)
                if close[i] > h3_12h[i] and close[i] > hma_50_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Camarilla L3 AND price < 1d HMA50 (bearish trend)
                elif close[i] < l3_12h[i] and close[i] < hma_50_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals