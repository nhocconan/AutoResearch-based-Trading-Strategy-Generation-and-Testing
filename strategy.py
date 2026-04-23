#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
- Long: Williams %R(14) crosses above -80 (oversold) AND price > 1d EMA50 AND volume > 2.0x 20-period avg
- Short: Williams %R(14) crosses below -20 (overbought) AND price < 1d EMA50 AND volume > 2.0x 20-period avg
- Exit: Opposite Williams %R cross OR price crosses 1d EMA50
- Uses 1d HTF for EMA50 (calculated from prior completed bars)
- Designed for low trade frequency (19-50/year) to minimize fee drag on 4h timeframe
- Williams %R provides mean reversion signals in ranging markets while trend filter avoids counter-trend trades
- Volume confirmation filters low-conviction moves
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 4h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need 50 for EMA, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Williams %R signals
        wr_cross_up = williams_r[i] > -80 and williams_r[i-1] <= -80  # Cross above -80 (oversold)
        wr_cross_down = williams_r[i] < -20 and williams_r[i-1] >= -20  # Cross below -20 (overbought)
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND price > 1d EMA50 AND volume confirmation
            if wr_cross_up and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < 1d EMA50 AND volume confirmation
            elif wr_cross_down and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 OR price < 1d EMA50 (trend flip)
            if wr_cross_down or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 OR price > 1d EMA50 (trend flip)
            if wr_cross_up or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0