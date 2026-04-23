#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
- Long: Williams %R(14) crosses above -80 (oversold bounce) AND price > 1d EMA50 AND volume > 2.0x 24-period avg
- Short: Williams %R(14) crosses below -20 (overbought rejection) AND price < 1d EMA50 AND volume > 2.0x 24-period avg
- Exit: Williams %R crosses below -50 (for longs) OR above -50 (for shorts) OR price crosses 1d EMA50
- Uses 1d HTF for EMA50 trend filter
- Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe
- Williams %R is effective in ranging/bear markets for mean reversion entries
- Volume confirmation reduces false reversals
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
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 24)  # Need 50 for EMA, 14 for Williams %R, 24 for volume MA
    
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
        # Long: Williams %R crosses above -80 (from below)
        wr_cross_up_80 = (williams_r[i] > -80) and (williams_r[i-1] <= -80)
        # Short: Williams %R crosses below -20 (from above)
        wr_cross_down_20 = (williams_r[i] < -20) and (williams_r[i-1] >= -20)
        # Exit signals: Williams %R crosses -50
        wr_cross_down_50 = (williams_r[i] < -50) and (williams_r[i-1] >= -50)  # Long exit
        wr_cross_up_50 = (williams_r[i] > -50) and (williams_r[i-1] <= -50)    # Short exit
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND price > 1d EMA50 AND volume confirmation
            if wr_cross_up_80 and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < 1d EMA50 AND volume confirmation
            elif wr_cross_down_20 and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR price < 1d EMA50 (trend flip)
            if wr_cross_down_50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR price > 1d EMA50 (trend flip)
            if wr_cross_up_50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0