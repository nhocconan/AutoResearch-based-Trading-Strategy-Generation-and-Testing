#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h trend filter (HMA21) and volume confirmation
# Uses Camarilla levels from 1d data: breakout above H3 = long, below L3 = short
# 12h HMA21 filter ensures trades align with higher timeframe trend
# Volume confirmation reduces false breakouts
# Designed for 4h timeframe to target 20-50 trades/year (75-200 over 4 years)
# Works in bull/bear: HMA21 adapts to trend, Camarilla provides structure

name = "4h_12h_camarilla_hma_volume_v2"
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
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Load 12h data ONCE before loop for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21 with proper min_periods
    close_12h = pd.Series(df_12h['close'].values)
    half_n = int(21/2 + 0.5)
    wma_half = close_12h.rolling(window=half_n, min_periods=half_n).mean()
    wma_full = close_12h.rolling(window=21, min_periods=21).mean()
    hma_21_12h = (2 * wma_half - wma_full).values
    
    # Align 12h HMA21 to 4h timeframe
    hma_21_4h = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
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
        if (np.isnan(camarilla_h3_4h[i]) or np.isnan(camarilla_l3_4h[i]) or
            np.isnan(hma_21_4h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 OR trend turns bearish
            if close[i] < camarilla_l3_4h[i] or close[i] < hma_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 OR trend turns bullish
            if close[i] > camarilla_h3_4h[i] or close[i] > hma_21_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Camarilla H3 AND price > 12h HMA21 (bullish trend)
                if close[i] > camarilla_h3_4h[i] and close[i] > hma_21_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Camarilla L3 AND price < 12h HMA21 (bearish trend)
                elif close[i] < camarilla_l3_4h[i] and close[i] < hma_21_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals