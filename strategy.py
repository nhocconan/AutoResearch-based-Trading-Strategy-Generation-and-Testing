#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation.
- Williams %R(14): measures overbought/oversold levels (-20 to -80 range)
- Long: Williams %R crosses above -80 from below + volume > 1.5x 20-period avg + price > 1d EMA34
- Short: Williams %R crosses below -20 from above + volume > 1.5x 20-period avg + price < 1d EMA34
- Exit: Opposite Williams %R crossover (-20 for long, -80 for short) or EMA34 trend flip
- Uses Williams %R for mean reversion signals, volume for conviction, 1d EMA34 for HTF trend filter
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Williams %R works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
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
    
    # Volume confirmation: > 1.5x 20-period average (balanced to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
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
        
        # Williams %R crossovers (using previous bar to detect actual cross)
        prev_williams_r = williams_r[i-1]
        
        if position == 0:
            # Long: Williams %R crosses above -80 + volume confirmation + price > 1d EMA34
            if (williams_r[i] > -80 and prev_williams_r <= -80 and 
                volume_confirm and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 + volume confirmation + price < 1d EMA34
            elif (williams_r[i] < -20 and prev_williams_r >= -20 and 
                  volume_confirm and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 OR price < 1d EMA34 (trend flip)
            if (williams_r[i] > -20 and prev_williams_r <= -20) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 OR price > 1d EMA34 (trend flip)
            if (williams_r[i] < -80 and prev_williams_r >= -80) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0