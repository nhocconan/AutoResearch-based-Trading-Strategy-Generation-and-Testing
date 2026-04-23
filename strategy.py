#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA50 trend filter and volume confirmation.
- Williams %R(14) measures overbought/oversold: < -80 = oversold, > -20 = overbought
- Long: Williams %R crosses above -80 from below (exit oversold) + price > 1d EMA50 + volume > 1.5x avg
- Short: Williams %R crosses below -20 from above (exit overbought) + price < 1d EMA50 + volume > 1.5x avg
- Exit: Williams %R crosses opposite extreme (-20 for long, -80 for short) OR trailing stop (2.5x ATR)
- Uses 1d EMA50 as trend filter to avoid counter-trend trades
- Volume confirmation reduces false reversals
- Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams %R(14) on primary timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # Need 14 for Williams %R/ATR, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Williams %R signals: crossing extremes
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        # Long signal: exits oversold (< -80) 
        long_signal = (wr_prev <= -80) and (wr > -80)
        # Short signal: exits overbought (> -20)
        short_signal = (wr_prev >= -20) and (wr < -20)
        
        # Exit signals: re-enter extreme zones
        long_exit = wr >= -20  # Re-enter overbought
        short_exit = wr <= -80  # Re-enter oversold
        
        if position == 0:
            # Long: exit oversold + price > 1d EMA50 + volume confirmation
            if long_signal and close[i] > ema_50_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: exit overbought + price < 1d EMA50 + volume confirmation
            elif short_signal and close[i] < ema_50_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme for trailing stop
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Williams %R re-enters overbought (exit signal)
            # 2. Trailing stop (2.5x ATR from extreme)
            williams_exit = long_exit
            trailing_stop = close[i] < long_extreme - 2.5 * atr[i]
            
            if williams_exit or trailing_stop:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme for trailing stop
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Williams %R re-enters oversold (exit signal)
            # 2. Trailing stop (2.5x ATR from extreme)
            williams_exit = short_exit
            trailing_stop = close[i] > short_extreme + 2.5 * atr[i]
            
            if williams_exit or trailing_stop:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0