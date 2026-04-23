#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1w EMA50 Trend and Volume Confirmation
- Elder Ray measures bull/bear power relative to EMA13 to detect momentum exhaustion
- 1w EMA(50) defines the major trend: only take longs above EMA50, shorts below
- Volume > 1.3x 20-period average confirms conviction in the move
- Designed for 6h timeframe targeting 12-30 trades/year (50-120 over 4 years) to minimize fee drag
- Works in bull markets via buy-the-dip on pullbacks to EMA13, in bear markets via sell-the-rally
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13  # negative values indicate bearish pressure
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # EMA1w, volume MA, EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray signals with 1w trend filter and volume confirmation
        # Long: bull power rising (momentum building) + above weekly EMA50 + volume confirmation
        # Short: bear power falling (increasing bearish pressure) + below weekly EMA50 + volume confirmation
        long_signal = (bull_power[i] > bull_power[i-1] and  # rising bull power
                      close[i] > ema_50_1w_aligned[i] and   # above weekly trend
                      volume[i] > 1.3 * vol_ma[i])
        
        short_signal = (bear_power[i] < bear_power[i-1] and  # falling bear power (more negative)
                       close[i] < ema_50_1w_aligned[i] and   # below weekly trend
                       volume[i] > 1.3 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: momentum reversal or trend violation
            exit_signal = False
            
            if position == 1:
                # Exit long: bull power weakening OR price breaks below weekly EMA50
                if (bull_power[i] < bull_power[i-1] or  # bull power declining
                    close[i] < ema_50_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: bear power weakening OR price breaks above weekly EMA50
                if (bear_power[i] > bear_power[i-1] or  # bear power declining (less negative)
                    close[i] > ema_50_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1wEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0