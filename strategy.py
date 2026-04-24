#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
- Long when Lips > Teeth > Jaw (bullish alignment) and close > 1d EMA50
- Short when Lips < Teeth < Jaw (bearish alignment) and close < 1d EMA50
- Volume must be > 1.5x 20-period average for confirmation
- Fixed position size: 0.25 long/short, 0.0 flat
- Uses 1d HTF for trend filter to avoid whipsaws in ranging markets
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
- Designed to work in both bull and bear markets via strong trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period SMA, smoothed by 8 periods
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA, smoothed by 5 periods
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA, smoothed by 3 periods
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.rolling(window=3, min_periods=3).mean().values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13+8, 8+5, 5+3, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish Alligator alignment (Lips > Teeth > Jaw) + uptrend + volume
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment (Lips < Teeth < Jaw) + downtrend + volume
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks down (Lips <= Teeth or Teeth <= Jaw)
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks up (Lips >= Teeth or Teeth >= Jaw)
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0