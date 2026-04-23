#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Mean Reversion with 1d EMA50 trend filter and volume spike confirmation.
- Long: Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume > 1.8x 24-period avg
- Short: Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume > 1.8x 24-period avg
- Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
- Uses 1d HTF for EMA50 and Williams %R (calculated from prior completed bars)
- Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag
- Williams %R identifies extreme reversals in ranging markets; EMA50 filters for trend alignment
- Volume confirmation ensures conviction behind moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.8x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R from prior 1d bar (HTF = 1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, 14)  # Need 50 for EMA, 24 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Williams %R signals (using prior completed 1d bar values)
        wr = williams_r_aligned[i-1]
        wr_prev = williams_r_aligned[i-2] if i >= 2 else wr
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume confirmation
            if wr < -80 and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume confirmation
            elif wr > -20 and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (momentum fading)
            if wr > -50 and wr_prev <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (momentum fading)
            if wr < -50 and wr_prev >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_MeanReversion_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0