#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with volume confirmation and 1w trend filter
# - Uses 1d Donchian channel (20) for breakout signals
# - Uses 1w EMA(50) to filter trend direction (only trade in direction of weekly trend)
# - Requires volume spike (2x 20-period average) for confirmation
# - Exits when price crosses back below/above the 10-period EMA on 4h
# - Designed to capture strong trending moves with multi-timeframe alignment
# - Target: 20-50 total trades over 4 years (5-12/year) with 0.30 position sizing

name = "4h_1dDonchian_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian Channel (20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high over last 20 days
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 days
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    donchian_upper_4h = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_4h = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Align 1w EMA to 4h timeframe
    ema_50_1w_4h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    # 4h EMA(10) for exit condition
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or 
            np.isnan(ema_50_1w_4h[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine trend direction from 1w EMA
            uptrend = close[i] > ema_50_1w_4h[i]
            downtrend = close[i] < ema_50_1w_4h[i]
            
            # Long: price breaks above 1d Donchian upper with volume spike in uptrend
            if uptrend and close[i] > donchian_upper_4h[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 1d Donchian lower with volume spike in downtrend
            elif downtrend and close[i] < donchian_lower_4h[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA(10)
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above 4h EMA(10)
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals