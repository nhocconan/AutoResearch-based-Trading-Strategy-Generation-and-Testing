#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion using Bollinger Bands with 4h trend filter and volume confirmation
# In ranging markets (common in 2025+), price reverts to mean at Bollinger Band extremes.
# 4h trend filter ensures we only take mean-reversion trades in the direction of higher timeframe momentum.
# Volume spike confirms institutional interest at the reversal point.
# Target: 60-150 total trades over 4 years (~15-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA trend filter (20-period)
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Bollinger Bands (20, 2) on 1h
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + bb_std * std_20
    bb_lower = sma_20 - bb_std * std_20
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need BB (20), volume MA (20), 4h EMA (20)
    start_idx = max(bb_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 4h EMA
        bullish_trend = ema_20_4h_aligned[i] > sma_20[i]  # 4h trend vs 1h mean
        bearish_trend = ema_20_4h_aligned[i] < sma_20[i]
        
        if position == 0:
            # Long: price at lower BB with volume and bullish 4h trend
            if price <= bb_lower[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price at upper BB with volume and bearish 4h trend
            elif price >= bb_upper[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle (mean reversion) or 4h trend turns bearish
            if price >= sma_20[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle (mean reversion) or 4h trend turns bullish
            if price <= sma_20[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Bollinger_MeanReversion_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0