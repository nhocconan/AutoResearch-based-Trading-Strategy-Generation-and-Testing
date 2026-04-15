#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d volume confirmation and 1w trend filter
# Works in bull markets (breakouts above upper band in uptrend) and bear markets (breakouts below lower band in downtrend)
# Uses Bollinger Bands (20,2) from 12h for squeeze detection, volume spike from 1d to confirm breakout strength,
# and 1w EMA50 for trend alignment. Targets low trade frequency (15-30/year) with clear trend-following logic.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Bollinger Bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Bollinger Bands (20,2) on 12h
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Bollinger Band width for squeeze detection (normalized by SMA)
    bb_width = (upper_band - lower_band) / sma_20
    
    # Squeeze condition: BB width below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    squeeze_aligned = align_htf_to_ltf(prices, df_12h, squeeze)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: Bollinger Band breakout above upper band + uptrend + volume squeeze release
        if (squeeze_aligned[i] and 
            close[i] > upper_band_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bollinger Band breakout below lower band + downtrend + volume squeeze release
        elif (squeeze_aligned[i] and 
              close[i] < lower_band_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price returns to SMA (mean reversion within trend)
        elif position == 1 and close[i] <= sma_20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= sma_20[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Bollinger_Squeeze_1dVolume_1wEMA_Trend"
timeframe = "12h"
leverage = 1.0