#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high AND price > weekly EMA50 AND volume > 1.5x daily average volume.
# Short when price breaks below 20-day low AND price < weekly EMA50 AND volume > 1.5x daily average volume.
# Exit when price crosses back below/above 10-day SMA (trailing exit).
# Uses Donchian for breakout structure, weekly EMA for trend filter, volume for confirmation.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
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
    
    # Get daily data for Donchian calculation (already daily, but using for consistency)
    df_1d = get_htf_data(prices, '1d')
    
    # Donchian channels (20-period high/low)
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    weekly_ema50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Get daily average volume for confirmation (20-day average)
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 10-day SMA for exit
    sma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure Donchian and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donchian_high = high_20[i]
        donchian_low = low_20[i]
        weekly_ema = weekly_ema50_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        vol = volume[i]
        sma = sma_10[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above 20-day high + weekly uptrend + volume confirmation
            if price > donchian_high and price > weekly_ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-day low + weekly downtrend + volume confirmation
            elif price < donchian_low and price < weekly_ema and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 10-day SMA
            if price < sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 10-day SMA
            if price > sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals