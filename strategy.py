#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with daily volume confirmation and weekly trend filter.
# Long when price breaks above Donchian(20) high AND daily volume > 1.5x average daily volume AND price > weekly EMA(50)
# Short when price breaks below Donchian(20) low AND daily volume > 1.5x average daily volume AND price < weekly EMA(50)
# Exit when price crosses back below/above Donchian midline (10-period average of high/low)
# Uses Donchian for trend/breakout structure, volume for confirmation, weekly EMA for higher timeframe trend filter.
# Target: 20-40 trades/year per symbol.
name = "4h_Donchian_Breakout_Volume_WeeklyTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Daily average volume (20-period)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Weekly EMA (50-period)
    weekly_ema50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Donchian channels (20-period high/low) - calculated on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure Donchian and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(weekly_ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = high_20[i]
        lower_channel = low_20[i]
        midline = donchian_mid[i]
        vol_ma = vol_ma_1d_aligned[i]
        weekly_ema = weekly_ema50_aligned[i]
        vol = volume[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: break above upper channel + volume confirmation + above weekly EMA
            if price > upper_channel and volume_confirmed and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower channel + volume confirmation + below weekly EMA
            elif price < lower_channel and volume_confirmed and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midline
            if price < midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midline
            if price > midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals