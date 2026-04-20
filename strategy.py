#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation.
# Works in bull markets (trend-following breakouts) and bear markets (short breakdowns).
# Volume filter reduces false breakouts. Uses 1d trend to align with higher timeframe momentum.
# Target: 20-40 trades/year per symbol to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20) for breakout signals
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h volume average (20) for volume confirmation
    volume_4h = df_4h['volume'].values
    vol_avg = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_4h, vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        volume_val = prices['volume'].iloc[i]
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_avg_val = vol_avg_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(ema_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band with 1d uptrend and volume confirmation
            if close_val > upper_val and close_val > ema_val and volume_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with 1d downtrend and volume confirmation
            elif close_val < lower_val and close_val < ema_val and volume_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below lower Donchian band
            if close_val < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper Donchian band
            if close_val > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0