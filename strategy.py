#!/usr/bin/env python3
"""
6h_ElderRay_BullPower_BearPower_1dTrend_Volume
Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) captures institutional buying/selling pressure. Combined with 1-day EMA50 trend filter and volume spike (>1.5x average) to confirm institutional participation. Works in bull/bear by following higher timeframe trend. Targets 20-30 trades/year via strict Elder Ray divergence conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close for Elder Ray calculation
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components on 1d data
    # Bull Power = High - EMA13
    bull_power = df_1d['high'].values - ema_13_1d
    # Bear Power = Low - EMA13
    bear_power = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: >1.5x 24-period MA (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_24[i])
        
        # Elder Ray signals: Bull Power rising + Bear Power weakening for long
        # Bear Power falling + Bull Power weakening for short
        bull_rising = bull_power_aligned[i] > bull_power_aligned[i-1]
        bear_falling = bear_power_aligned[i] < bear_power_aligned[i-1]
        bull_falling = bull_power_aligned[i] < bull_power_aligned[i-1]
        bear_rising = bear_power_aligned[i] > bear_power_aligned[i-1]
        
        # Long: Bull Power rising AND Bear Power falling (bulls gaining control)
        long_signal = bull_rising and bear_falling and vol_confirm and uptrend
        # Short: Bear Power rising AND Bull Power falling (bears gaining control)
        short_signal = bear_rising and bull_falling and vol_confirm and downtrend
        
        # Exit: when Elder Ray divergence fails or trend changes
        long_exit = not (bull_rising and bear_falling) or not uptrend
        short_exit = not (bear_rising and bull_falling) or not downtrend
        
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_BullPower_BearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0