#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyTrend_Donchian20_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Donchian channel on 6h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need weekly EMA and daily volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_1d_aligned[i]
        upper = high_20[i]
        lower = low_20[i]
        weekly_trend = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.8x daily average
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: Price breaks above upper Donchian band in uptrend with volume
            if price > upper and price > weekly_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band in downtrend with volume
            elif price < lower and price < weekly_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price closes below lower Donchian band
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price closes above upper Donchian band
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals