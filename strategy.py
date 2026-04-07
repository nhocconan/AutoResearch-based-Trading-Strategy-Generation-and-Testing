#!/usr/bin/env python3
"""
1d_aroon_oscillator_1w_trend_volume_v1
Hypothesis: Aroon Oscillator (Aroon Up - Aroon Down) identifies trend strength on weekly timeframe.
When combined with daily price breaks above/below weekly Aroon-based bands and volume confirmation,
it captures strong trend continuations in both bull and bear markets. Weekly trend filter prevents
counter-trend trades. Targets 7-25 trades/year by requiring Aroon signal + price break + volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_aroon_oscillator_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly OHLC for Aroon calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    # Calculate Aroon Up and Down (25-period)
    high_weekly = df_1w['high'].values
    low_weekly = df_1w['low'].values
    close_weekly = df_1w['close'].values
    
    aroon_up = np.full(len(high_weekly), np.nan)
    aroon_down = np.full(len(low_weekly), np.nan)
    
    for i in range(25, len(high_weekly)):
        # Periods since highest high
        highest_high_idx = i - np.argmax(high_weekly[i-24:i+1])  # 25-period lookback
        aroon_up[i] = ((25 - highest_high_idx) / 25) * 100
        
        # Periods since lowest low
        lowest_low_idx = i - np.argmin(low_weekly[i-24:i+1])
        aroon_down[i] = ((25 - lowest_low_idx) / 25) * 100
    
    aroon_osc = aroon_up - aroon_down  # -100 to +100
    
    # Weekly Bollinger Bands (20, 2) for dynamic support/resistance
    sma20 = pd.Series(close_weekly).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_weekly).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    # Align weekly indicators to daily timeframe
    aroon_osc_aligned = align_htf_to_ltf(prices, df_1w, aroon_osc)
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Daily 50-period volume average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(aroon_osc_aligned[i]) or 
            np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower band OR Aroon turns negative
            if close[i] < lower_band_aligned[i] or aroon_osc_aligned[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above upper band OR Aroon turns positive
            if close[i] > upper_band_aligned[i] or aroon_osc_aligned[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Aroon bullish (>50) + price breaks above upper band + volume
            if (aroon_osc_aligned[i] > 50 and 
                close[i] > upper_band_aligned[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short: Aroon bearish (<-50) + price breaks below lower band + volume
            elif (aroon_osc_aligned[i] < -50 and 
                  close[i] < lower_band_aligned[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals