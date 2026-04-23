#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme Reversal with 1d EMA50 trend filter and volume spike confirmation.
- Williams %R(14): Long when < -80 (oversold) and turning up, Short when > -20 (overbought) and turning down
- Trend filter: price > 1d EMA50 for longs, price < 1d EMA50 for shorts
- Volume confirmation: current volume > 2.0 x 20-period average volume
- Exit: Williams %R crosses back through -50 (mean reversion completion) or ATR trailing stop (2.0x ATR)
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 12-37 trades/year (50-150 total over 4 years) on 12h timeframe to avoid overtrading
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 2.0x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 14, 50)  # Need 20 for volume MA, 14 for ATR/Williams %R, 50 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R momentum conditions
        williams_rising = williams_r[i] > williams_r[i-1]  # Williams %R rising (less negative)
        williams_falling = williams_r[i] < williams_r[i-1]  # Williams %R falling (more negative)
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + rising + price > 1d EMA50 + volume spike
            if williams_r[i] < -80 and williams_rising and close[i] > ema_50_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + falling + price < 1d EMA50 + volume spike
            elif williams_r[i] > -20 and williams_falling and close[i] < ema_50_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions:
            # 1. Williams %R crosses back above -50 (mean reversion completion)
            # 2. Price reverses 2.0x ATR from entry (trailing stop)
            mean_reversion_exit = williams_r[i] > -50
            # Approximate trailing stop using close-based condition
            trailing_stop_long = close[i] < close[i-1] - 2.0 * atr[i]  # Simplified close-based stop
            
            if mean_reversion_exit or trailing_stop_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions:
            # 1. Williams %R crosses back below -50 (mean reversion completion)
            # 2. Price reverses 2.0x ATR from entry (trailing stop)
            mean_reversion_exit = williams_r[i] < -50
            # Approximate trailing stop using close-based condition
            trailing_stop_short = close[i] > close[i-1] + 2.0 * atr[i]  # Simplified close-based stop
            
            if mean_reversion_exit or trailing_stop_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dEMA50_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0