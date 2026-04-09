#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 6h EMA filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13 - works in both bull and bear markets
# Long when Bull Power > 0 and Bear Power < 0 (strong bullish momentum)
# Short when Bear Power > 0 and Bull Power < 0 (strong bearish momentum)
# Volume confirmation filters weak breakouts
# EMA21 on 6h ensures trading with intermediate trend
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray components
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Bear Power = Low - EMA13
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h EMA21 for trend filter
    ema21_6h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(ema21_6h[i]) or np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.2x average 6h volume
        volume_confirmed = volume[i] > 1.2 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when bear power turns positive or price closes below EMA21
            if bear_power_1d_aligned[i] > 0 or close[i] < ema21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when bull power turns positive or price closes above EMA21
            if bull_power_1d_aligned[i] > 0 or close[i] > ema21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Elder Ray signals with volume confirmation
            # Strong bullish: Bull Power > 0 AND Bear Power < 0
            # Strong bearish: Bear Power > 0 AND Bull Power < 0
            if volume_confirmed:
                if bull_power_1d_aligned[i] > 0 and bear_power_1d_aligned[i] < 0:
                    position = 1
                    signals[i] = position_size
                elif bear_power_1d_aligned[i] > 0 and bull_power_1d_aligned[i] < 0:
                    position = -1
                    signals[i] = -position_size
    
    return signals