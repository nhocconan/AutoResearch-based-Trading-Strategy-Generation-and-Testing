#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakout with 1d EMA trend filter and volume confirmation
# - Uses 1w Donchian channel (20) for breakout signals
# - Uses 1d EMA(50) to filter trend direction (only trade in direction of trend)
# - Requires volume spike (2x 20-period average) for confirmation
# - Exits when price crosses back below/above 1d EMA(50) or opposite Donchian breakout occurs
# - Designed to capture strong trends with proper filtering to avoid whipsaws
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_1wDonchian_1dEMA50_Volume_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w Donchian Channel (20 periods)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper and lower bands
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w indicators to 1d timeframe
    donchian_high_1d = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Align 1d EMA to 1d timeframe (no alignment needed as already 1d)
    ema_50_1d = ema_50
    
    # Volume filters (1d timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian high with volume spike and price above 1d EMA50
            if close[i] > donchian_high_1d[i] and volume_spike[i] and close[i] > ema_50_1d[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian low with volume spike and price below 1d EMA50
            elif close[i] < donchian_low_1d[i] and volume_spike[i] and close[i] < ema_50_1d[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 OR price breaks below 1w Donchian low
            if close[i] < ema_50_1d[i] or close[i] < donchian_low_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 OR price breaks above 1w Donchian high
            if close[i] > ema_50_1d[i] or close[i] > donchian_high_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals