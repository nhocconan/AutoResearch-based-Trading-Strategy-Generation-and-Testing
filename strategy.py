#!/usr/bin/env python3
"""
1h_HMA21_Cross_4hTrend_1dVolumeFilter
Strategy: 1h HMA(21) cross with 4h trend filter and 1d volume spike confirmation.
Long: HMA(21) crosses above price + 4h close > 4h HMA(21) + 1d volume > 1.5x 20-day avg
Short: HMA(21) crosses below price + 4h close < 4h HMA(21) + 1d volume > 1.5x 20-day avg
Exit: Opposite HMA cross
Position size: 0.20
Uses HMA for smooth trend following, 4h for trend direction, 1d volume for conviction.
Avoids whipsaws by requiring alignment across timeframes and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def hull_moving_average(series, period):
    """Calculate Hull Moving Average"""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(series).rolling(window=half_period, min_periods=half_period).mean()
    wma2 = pd.Series(series).rolling(window=period, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    return hma.values

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
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate HMA(21) on 1h
    hma_21 = hull_moving_average(close, 21)
    
    # Calculate HMA(21) on 4h for trend filter
    hma_21_4h = hull_moving_average(close_4h, 21)
    
    # Calculate 20-day average volume on 1d
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1h timeframe
    hma_21_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_21_4h)
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for HMA calculations
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(hma_21[i]) or np.isnan(hma_21_4h_aligned[i]) or 
            np.isnan(volume_ma20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume aligned to 1h
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_filter = vol_1d_current > (1.5 * volume_ma20_1d_aligned[i])
        
        # Trend filter: 4h close vs 4h HMA(21)
        uptrend_4h = close_4h[-1] > hma_21_4h[-1] if len(close_4h) > 0 else False
        downtrend_4h = close_4h[-1] < hma_21_4h[-1] if len(close_4h) > 0 else False
        
        # HMA cross signals
        hma_cross_up = close[i] > hma_21[i] and close[i-1] <= hma_21[i-1]
        hma_cross_down = close[i] < hma_21[i] and close[i-1] >= hma_21[i-1]
        
        if position == 0:
            # Long: HMA cross up + 4h uptrend + volume spike
            if hma_cross_up and uptrend_4h and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short: HMA cross down + 4h downtrend + volume spike
            elif hma_cross_down and downtrend_4h and volume_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: HMA cross down
            if hma_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: HMA cross up
            if hma_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_HMA21_Cross_4hTrend_1dVolumeFilter"
timeframe = "1h"
leverage = 1.0