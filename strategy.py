#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 1d EMA34 trend filter and volume spike.
- Williams %R(14) from prior 1d: oversold < -80 (long), overbought > -20 (short)
- Long: %R crosses above -80 + volume > 2.0x 20-period avg + price > 1d EMA34
- Short: %R crosses below -20 + volume > 2.0x 20-period avg + price < 1d EMA34
- Exit: %R crosses above -50 (for long) or below -50 (for short) OR trend flip
- Uses Williams %R for mean reversion signals, volume for conviction, 1d EMA34 for HTF filter
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)
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
    
    # Volume confirmation: > 2.0x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) from prior 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values  # Prior 1d close
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback on 1d data
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d_prev) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 4h timeframe (no extra delay needed as it's based on completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20, 14)  # Need 50 for EMA34, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Williams %R levels
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) + volume confirmation + price > 1d EMA34
            if (wr > -80 and wr_prev <= -80 and 
                volume_confirm and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) + volume confirmation + price < 1d EMA34
            elif (wr < -20 and wr_prev >= -20 and 
                  volume_confirm and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (momentum fade) OR price < 1d EMA34 (trend flip)
            if (wr > -50 and wr_prev <= -50) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (momentum fade) OR price > 1d EMA34 (trend flip)
            if (wr < -50 and wr_prev >= -50) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0