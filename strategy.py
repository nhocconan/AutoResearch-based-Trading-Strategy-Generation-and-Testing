#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band Breakout with 1-week EMA Trend Filter and Volume Confirmation
# Uses Bollinger Bands (20, 2) on daily timeframe for breakout signals.
# Trend filter: price above/below 50-period EMA on weekly timeframe.
# Volume confirmation: current day's volume > 1.5x 20-day average volume.
# Works in bull markets (breakouts above upper band) and bear markets (breakouts below lower band).
# Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Load weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20, 2) on daily
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Calculate 50-period EMA on weekly
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Bollinger Bands to daily timeframe (no shift needed as they are for same day)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Align weekly EMA to daily timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 20-day average volume on daily
    avg_vol_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(avg_vol_20_aligned[i])):
            continue
        
        # Long entry: price breaks above upper Bollinger Band + price above weekly EMA + volume confirmation
        if (close[i] > upper_band_aligned[i] and
            close[i] > ema_50_aligned[i] and
            volume[i] > 1.5 * avg_vol_20_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower Bollinger Band + price below weekly EMA + volume confirmation
        elif (close[i] < lower_band_aligned[i] and
              close[i] < ema_50_aligned[i] and
              volume[i] > 1.5 * avg_vol_20_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or price crosses back below/above weekly EMA
        elif position == 1 and (close[i] < ema_50_aligned[i] or close[i] < lower_band_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_50_aligned[i] or close[i] > upper_band_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_Bollinger_Breakout_1wEMA_Volume"
timeframe = "1d"
leverage = 1.0