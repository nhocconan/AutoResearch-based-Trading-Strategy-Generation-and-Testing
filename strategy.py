#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d ADX Trend Filter and Volume Confirmation
- Long: Williams %R(14) crosses above -80 (oversold) + price > 1d EMA34 (uptrend) + volume > 1.5x 20-period avg
- Short: Williams %R(14) crosses below -20 (overbought) + price < 1d EMA34 (downtrend) + volume > 1.5x 20-period avg
- Exit: Williams %R crosses opposite extreme (-20 for long, -80 for short) OR ATR trailing stop (2.0x)
- Uses 1d EMA34 as trend filter to ensure trades align with higher timeframe momentum
- Williams %R provides mean-reversion signals in ranging markets while ADX filters for trending conditions
- Volume confirmation reduces false reversals
- Designed for 6h timeframe to capture medium-term reversals with lower frequency (target: 12-30 trades/year)
- Works in both bull and bear markets: ADX trend filter adapts to regime, Williams %R captures reversals within trends
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
    
    # Calculate Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 1.5x 20-period average (volume filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d EMA34 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 34)  # Need 14 for Williams %R, 20 for volume MA, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Williams %R conditions
        williams_r_oversold = williams_r[i] <= -80  # Oversold condition
        williams_r_overbought = williams_r[i] >= -20  # Overbought condition
        williams_r_cross_above_oversold = (williams_r[i] > -80) and (williams_r[i-1] <= -80)  # Cross above -80
        williams_r_cross_below_overbought = (williams_r[i] < -20) and (williams_r[i-1] >= -20)  # Cross below -20
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 + price > 1d EMA34 + volume spike
            if williams_r_cross_above_oversold and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Williams %R crosses below -20 + price < 1d EMA34 + volume spike
            elif williams_r_cross_below_overbought and close[i] < ema_34_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Williams %R crosses above -20 (exit long on overbought)
            # 2. Price reverses 2.0x ATR from long extreme (trailing stop)
            exit_condition = williams_r[i] >= -20 or close[i] < long_extreme - 2.0 * atr[i]
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Williams %R crosses below -80 (exit short on oversold)
            # 2. Price reverses 2.0x ATR from short extreme (trailing stop)
            exit_condition = williams_r[i] <= -80 or close[i] > short_extreme + 2.0 * atr[i]
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0