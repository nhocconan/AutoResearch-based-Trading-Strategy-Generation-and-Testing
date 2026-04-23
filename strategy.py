#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Reversal with 1d EMA34 trend filter and volume spike.
- Williams %R(14): momentum oscillator identifying overbought/oversold conditions
- Long: Williams %R crosses above -80 (from oversold) + volume > 1.8x 20-period avg + price > 1d EMA34
- Short: Williams %R crosses below -20 (from overbought) + volume > 1.8x 20-period avg + price < 1d EMA34
- Exit: Williams %R returns to opposite extreme (-20 for long, -80 for short) OR 1d EMA34 trend flip
- Uses Williams %R for mean reversion signals, volume for conviction, 1d EMA34 for HTF trend filter
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy reversals in uptrend) and bear (sell reversals in downtrend)
- Williams %R is effective in ranging markets and captures reversals at extremes
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
    
    # Volume confirmation: > 1.8x 20-period average (balanced to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 1d timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero (when high == low)
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20, 14)  # Need 50 for safety, 34 for EMA, 20 for volume, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from oversold) + volume confirmation + price > 1d EMA34
            if (williams_r_aligned[i] > -80 and 
                williams_r_aligned[i-1] <= -80 and  # Cross above -80
                volume_confirm and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from overbought) + volume confirmation + price < 1d EMA34
            elif (williams_r_aligned[i] < -20 and 
                  williams_r_aligned[i-1] >= -20 and  # Cross below -20
                  volume_confirm and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -20 (overbought) OR price < 1d EMA34 (trend flip)
            if williams_r_aligned[i] >= -20 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -80 (oversold) OR price > 1d EMA34 (trend flip)
            if williams_r_aligned[i] <= -80 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0