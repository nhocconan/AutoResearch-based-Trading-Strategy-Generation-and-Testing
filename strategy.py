#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation.
- Williams %R(14): overbought > -20, oversold < -80
- Long: %R crosses above -80 from below + volume > 1.5x 20-period avg + price > 1d EMA34
- Short: %R crosses below -20 from above + volume > 1.5x 20-period avg + price < 1d EMA34
- Exit: %R crosses above -50 (for longs) or below -50 (for shorts) OR EMA34 trend flip
- Uses %R for mean reversion in extremes, volume for conviction, 1d EMA34 for HTF filter
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend)
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
    
    # Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need 34 for EMA34, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Williams %R reversal signals
            # Long: %R crosses above -80 from below (oversold reversal)
            wr_cross_up = williams_r[i] > -80 and williams_r[i-1] <= -80
            # Short: %R crosses below -20 from above (overbought reversal)
            wr_cross_down = williams_r[i] < -20 and williams_r[i-1] >= -20
            
            if wr_cross_up and volume_confirm and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif wr_cross_down and volume_confirm and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: %R crosses above -50 (momentum fading) OR EMA34 trend flip
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: %R crosses below -50 (momentum fading) OR EMA34 trend flip
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0