#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d EMA50 trend filter and volume confirmation.
- Long when price breaks above Camarilla R3 level AND close > 1d EMA50 (bullish trend)
- Short when price breaks below Camarilla S3 level AND close < 1d EMA50 (bearish trend)
- Volume must be > 2.0x 20-period average for confirmation (tight filter)
- Uses 6h primary timeframe with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Camarilla pivots provide institutional support/resistance levels that work in ranging and trending markets
- EMA50 trend filter ensures alignment with higher timeframe momentum to avoid counter-trend trades
- Volume confirmation reduces false breakouts
- Designed for BTC/ETH with edge in both bull (breakout continuation) and bear (mean reversion at extremes) regimes
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
    
    # Calculate Camarilla levels using previous period (no look-ahead)
    # Camarilla: based on previous period's range
    prev_close = pd.Series(close).shift(1).values
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels: Close ± (Range * multiplier)
    camarilla_r3 = prev_close + (prev_range * 1.1 / 4)  # R3 = C + (H-L)*1.1/4
    camarilla_s3 = prev_close - (prev_range * 1.1 / 4)  # S3 = C - (H-L)*1.1/4
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3, trend up (close > EMA50), volume confirmation
            if close[i] > camarilla_r3[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, trend down (close < EMA50), volume confirmation
            elif close[i] < camarilla_s3[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla S3 (mean reversion) OR trend reverses
            if close[i] < camarilla_s3[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla R3 (mean reversion) OR trend reverses
            if close[i] > camarilla_r3[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0