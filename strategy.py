#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter_VolumeSpike
Hypothesis: Trade 6h Elder Ray Bull/Bear Power crossovers with 1d EMA50 trend filter and volume confirmation (>1.8x 30-bar MA). 
Bull Power = High - EMA13, Bear Power = EMA13 - Low. Long when Bull Power crosses above zero AND 1d trend bullish. 
Short when Bear Power crosses above zero AND 1d trend bearish. Volume confirmation ensures breakout conviction. 
6h timeframe targets 12-37 trades/year to minimize fee drag. Discrete sizing 0.25 balances profit and fee drag. 
Works in bull/bear: 1d EMA50 filter adapts to market direction, Elder Ray captures momentum shifts, 
volume filters false breakouts. Uses discrete position sizing (0.0, ±0.25) to reduce churn.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA50 (50) and EMA13 (13) and volume MA (30)
    start_idx = max(50, 13, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power crosses above zero AND 1d trend bullish (close > EMA50) AND volume confirm
            long_setup = (bull_power[i] > 0) and (bull_power[i-1] <= 0) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_confirm[i]
            # Short: Bear Power crosses above zero AND 1d trend bearish (close < EMA50) AND volume confirm
            short_setup = (bear_power[i] > 0) and (bear_power[i-1] <= 0) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Bull Power crosses below zero OR 1d trend turns bearish
            if (bull_power[i] < 0) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bear Power crosses below zero OR 1d trend turns bullish
            if (bear_power[i] < 0) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0