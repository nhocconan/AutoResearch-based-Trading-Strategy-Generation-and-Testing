#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and daily trend filter.
# Long when price breaks above 4h Donchian upper band AND volume > 1.5x daily average volume AND price > daily EMA50 (bullish trend).
# Short when price breaks below 4h Donchian lower band AND volume > 1.5x daily average volume AND price < daily EMA50 (bearish trend).
# Exit when price crosses back below/above Donchian middle band (20-period average of high/low).
# Uses Donchian for breakout structure, volume for confirmation, daily EMA for trend filter.
# Target: 20-50 trades/year per symbol.

name = "4h_Donchian_Breakout_Volume_Trend"
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
    
    # Get daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 4h data (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_20 + low_20) / 2
    
    # Calculate daily EMA50 for trend filter
    daily_ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Calculate daily average volume (20-period) for confirmation
    daily_vol_avg = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    daily_vol_avg_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(daily_ema50_aligned[i]) or np.isnan(daily_vol_avg_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = high_20[i]
        lower_band = low_20[i]
        middle_band = donchian_middle[i]
        daily_ema = daily_ema50_aligned[i]
        daily_vol = daily_vol_avg_aligned[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x daily average
        volume_confirmed = vol > 1.5 * daily_vol
        
        if position == 0:
            # Long entry: price breaks above upper band AND bullish trend AND volume confirmation
            if price > upper_band and price > daily_ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band AND bearish trend AND volume confirmation
            elif price < lower_band and price < daily_ema and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below middle band
            if price < middle_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above middle band
            if price > middle_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals