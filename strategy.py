#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
- Long: Williams %R(14) < -80 (oversold) AND close > 1d EMA50 AND volume > 1.5x 20-period avg
- Short: Williams %R(14) > -20 (overbought) AND close < 1d EMA50 AND volume > 1.5x 20-period avg
- Exit: Williams %R crosses above -50 (long) or below -50 (short) OR trend flip (close crosses 1d EMA50)
- Uses 1d HTF for EMA50 (calculated from prior completed 1d bar)
- Designed for moderate trade frequency (30-60/year) to balance opportunity and fee drag on 4h timeframe
- Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)
- Volume confirmation reduces false signals
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 4h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # Need 50 for EMA, 14 for Williams %R, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 1d EMA50 AND volume confirmation
            if williams_r[i] < -80 and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < 1d EMA50 AND volume confirmation
            elif williams_r[i] > -20 and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 OR price < 1d EMA50 (trend flip)
            if williams_r[i] > -50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 OR price > 1d EMA50 (trend flip)
            if williams_r[i] < -50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0