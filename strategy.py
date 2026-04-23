#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation
- Elder Ray measures bull/bear power relative to EMA(13) to identify institutional buying/selling pressure
- 1d EMA(50) ensures alignment with higher timeframe trend to reduce counter-trend trades
- Volume spike (>1.8x 24-period average) confirms strong participation
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in both bull and bear markets by trading with the daily trend when institutional pressure aligns
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA(13) for Elder Ray (primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull power: high minus EMA
    bear_power = low - ema_13   # Bear power: low minus EMA
    
    # Volume confirmation: > 1.8x 24-period average (6h * 24 = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, 13)  # EMA1d, volume MA, EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray signals with trend filter
        # Long: strong bull power (> 0) + uptrend + volume spike
        # Short: strong bear power (< 0) + downtrend + volume spike
        long_signal = (bull_power[i] > 0 and 
                      close[i] > ema_50_1d_aligned[i] and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (bear_power[i] < 0 and 
                       close[i] < ema_50_1d_aligned[i] and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Elder Ray divergence or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: bear power turns positive (selling pressure) or trend reversal
                if (bear_power[i] > 0 or 
                    close[i] < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: bull power turns negative (buying pressure) or trend reversal
                if (bull_power[i] < 0 or 
                    close[i] > ema_50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0