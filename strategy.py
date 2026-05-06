#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND 1d close > 1d EMA34 AND volume > 1.5 * 20-bar average volume
# Short when price breaks below Camarilla S3 AND 1d close < 1d EMA34 AND volume > 1.5 * 20-bar average volume
# Exit when price retests the Camarilla H5/L5 level (mean of H3/L3 and H4/L4)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla levels provide robust intraday support/resistance structure
# 1d EMA34 filters for higher timeframe trend alignment
# Volume confirmation reduces false breakouts during low participation
# Works in both bull and bear markets by following the 1d trend

name = "12h_Camarilla_R3S3_1dEMA34_Volume_v1"
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
    
    # Calculate 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    # H5 = close + 2.0*(high-low), L5 = close - 2.0*(high-low)
    # H6 = close + 2.5*(high-low), L6 = close - 2.5*(high-low)
    # We use R3=H3, S3=L3, H5, L5
    daily_range = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * daily_range
    camarilla_l3 = close_1d - 1.1 * daily_range
    camarilla_h5 = close_1d + 2.0 * daily_range
    camarilla_l5 = close_1d - 2.0 * daily_range
    
    # Align HTF indicators to 12h timeframe (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Get 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_l5_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > H3 AND uptrend AND volume confirmation
            if close[i] > camarilla_h3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < L3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests H5 level from below
            if close[i] >= camarilla_h5_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests L5 level from above
            if close[i] <= camarilla_l5_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals