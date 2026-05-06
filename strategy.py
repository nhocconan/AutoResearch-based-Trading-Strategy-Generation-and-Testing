#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout with 1w trend filter and volume confirmation
# - Uses 1d Donchian channel (20-period high/low) for breakout signals
# - Uses 1w EMA (50-period) for trend direction filter
# - Requires volume spike (2x 20-period average) for confirmation
# - Exits when price crosses back below/above the 10-period EMA on 1d
# - Designed to capture strong trending moves with institutional volume
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_DonchianBreakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel and EMA exit
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian Channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band (20-period high)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band (20-period low)
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA (10-period) for exit
    close_1d = df_1d['close'].values
    ema_10 = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1w EMA (50-period) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_10_12h = align_htf_to_ltf(prices, df_1d, ema_10)
    
    # Align 1w EMA to 12h timeframe
    ema_50_1w_12h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filters (12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(ema_10_12h[i]) or np.isnan(ema_50_1w_12h[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout with volume spike and trend alignment
            # Long: price breaks above Donchian high with volume spike and price > 1w EMA50
            if close[i] > donchian_high_12h[i] and volume_spike[i] and close[i] > ema_50_1w_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike and price < 1w EMA50
            elif close[i] < donchian_low_12h[i] and volume_spike[i] and close[i] < ema_50_1w_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 10-period EMA on 1d
            if close[i] < ema_10_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 10-period EMA on 1d
            if close[i] > ema_10_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals