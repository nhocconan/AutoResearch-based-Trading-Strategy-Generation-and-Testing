#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band breakout with 12-hour EMA trend filter and volume confirmation
# Uses 4h Bollinger Bands for volatility-based entry signals, 12h EMA for trend direction,
# and volume spike to confirm breakout strength. Works in both bull and bear by
# only taking breakouts in the direction of the 12h trend. Designed for moderate
# trade frequency (target: 50-150 trades over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for Bollinger Bands and price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate Bollinger Bands (20, 2.0) on 4h
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    
    # Calculate EMA50 on 12h for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 4h)
    vol_avg_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_4h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_4h, lower_band)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_avg_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_4h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above upper Bollinger Band + volume spike + price above 12h EMA50
        if (close[i] > upper_band_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            close[i] > ema50_12h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower Bollinger Band + volume spike + price below 12h EMA50
        elif (close[i] < lower_band_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              close[i] < ema50_12h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price returns to middle Bollinger Band (SMA20)
        elif position == 1 and close[i] < sma_20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > sma_20[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Bollinger_12hEMA_Volume_Filter"
timeframe = "4h"
leverage = 1.0