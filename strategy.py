#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band breakout with 1w trend filter and volume spike confirmation.
# Long when price breaks above upper Bollinger Band (20, 2) AND 1w EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below lower Bollinger Band (20, 2) AND 1w EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Bollinger Band (between upper and lower bands).
# This strategy captures breakouts with higher timeframe trend alignment and volume confirmation to avoid false breakouts.
# Bollinger Bands provide dynamic volatility-based breakout levels. The 1w EMA50 filter ensures we trade with the higher timeframe trend.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_BollingerBreakout_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Bollinger Bands and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20-period, 2 std) from 1w data
    close_1w = df_1w['close'].values
    sma20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + (2 * std20)
    lower_band = sma20 - (2 * std20)
    
    # Align Bollinger Bands to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w EMA50 direction
    ema50_rising = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1w_aligned[1:] > ema50_1w_aligned[:-1]
    ema50_falling[1:] = ema50_1w_aligned[1:] < ema50_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average (using 12h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup for EMA50 and Bollinger Bands
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Bollinger Band, 1w EMA50 rising, volume filter
            long_cond = (close[i] > upper_band_aligned[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower Bollinger Band, 1w EMA50 falling, volume filter
            short_cond = (close[i] < lower_band_aligned[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below lower Bollinger Band
            if close[i] < lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above upper Bollinger Band
            if close[i] > upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals