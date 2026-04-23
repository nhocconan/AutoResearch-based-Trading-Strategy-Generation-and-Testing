#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 Trend Filter and Volume Confirmation
- Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
- Strong bull power + price above 12h EMA50 + volume spike = long entry
- Strong bear power + price below 12h EMA50 + volume spike = short entry
- Exit when power weakens or price crosses EMA50
- Works in bull markets via trend continuation, in bear markets via counter-trend power exhaustion
- Designed for 6h timeframe targeting 12-30 trades/year (50-120 over 4 years) to minimize fee drag
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
    open_time = prices['open_time'].values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13  # negative values indicate bear power
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 50, 20)  # EMA13, EMA50_12h, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray signals with trend filter and volume confirmation
        # Long: strong bull power + price above 12h EMA50 + volume spike
        # Short: strong bear power + price below 12h EMA50 + volume spike
        long_signal = (bull_power[i] > 0 and  # bull power positive
                      close[i] > ema_50_12h_aligned[i] and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (bear_power[i] < 0 and  # bear power negative (strong bear)
                       close[i] < ema_50_12h_aligned[i] and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: power weakening or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: bull power turns negative or price below EMA50
                if (bull_power[i] <= 0 or 
                    close[i] < ema_50_12h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: bear power turns positive or price above EMA50
                if (bear_power[i] >= 0 or 
                    close[i] > ema_50_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0