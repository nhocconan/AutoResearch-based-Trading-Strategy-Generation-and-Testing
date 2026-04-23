#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d EMA34 trend filter and volume spike confirmation.
- Long: Williams %R(14) crosses above -80 (oversold) + price > 1d EMA34 + volume > 1.8x 20-period avg volume
- Short: Williams %R(14) crosses below -20 (overbought) + price < 1d EMA34 + volume > 1.8x 20-period avg volume
- Exit: trailing stop (2.5x ATR from extreme) OR Williams %R crosses opposite extreme (-20 for long, -80 for short)
- Uses 1d EMA34 as trend filter for better regime adaptation
- Volume confirmation reduces false reversals
- ATR trailing stop manages risk
- Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag
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
    
    # Volume confirmation: > 1.8x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams %R(14) calculation
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 34)  # Need 20 for volume MA, 14 for ATR/Williams %R, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Williams %R extreme crossover conditions
        williams_r_prev = williams_r[i-1]
        williams_r_curr = williams_r[i]
        
        # Long signal: Williams %R crosses above -80 from below (oversold reversal)
        williams_r_long_signal = (williams_r_prev <= -80) and (williams_r_curr > -80)
        # Short signal: Williams %R crosses below -20 from above (overbought reversal)
        williams_r_short_signal = (williams_r_prev >= -20) and (williams_r_curr < -20)
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R oversold reversal + price > 1d EMA34 + volume spike
            if williams_r_long_signal and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Williams %R overbought reversal + price < 1d EMA34 + volume spike
            elif williams_r_short_signal and close[i] < ema_34_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from long extreme (trailing stop)
            # 2. Williams %R crosses below -20 (overbought)
            trailing_stop_long = close[i] < long_extreme - 2.5 * atr[i]
            williams_r_exit = (williams_r_prev >= -20) and (williams_r_curr < -20)
            
            if trailing_stop_long or williams_r_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from short extreme (trailing stop)
            # 2. Williams %R crosses above -80 (oversold)
            trailing_stop_short = close[i] > short_extreme + 2.5 * atr[i]
            williams_r_exit = (williams_r_prev <= -80) and (williams_r_curr > -80)
            
            if trailing_stop_short or williams_r_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0