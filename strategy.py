#!/usr/bin/env python3
# 6h_1d_supertrend_volume_v1
# Strategy: 6h Supertrend with 1d trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Supertrend captures trend direction with dynamic ATR-based bands.
# 1d Supertrend confirms higher timeframe trend to avoid counter-trend trades.
# Volume > 1.3x 20-period average confirms institutional participation.
# Designed for low trade frequency (~15-30/year) to minimize fee drag.
# Works in bull markets via long entries and bear markets via short entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # Average True Range
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close)
    final_lb = np.zeros_like(close)
    
    for i in range(len(close)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend_val = np.zeros_like(close)
    for i in range(len(close)):
        if i == 0:
            supertrend_val[i] = final_lb[i]
        else:
            if supertrend_val[i-1] == final_ub[i-1]:
                if close[i] <= final_ub[i]:
                    supertrend_val[i] = final_ub[i]
                else:
                    supertrend_val[i] = final_lb[i]
            else:
                if close[i] >= final_lb[i]:
                    supertrend_val[i] = final_lb[i]
                else:
                    supertrend_val[i] = final_ub[i]
    
    # Trend direction: 1 for uptrend, -1 for downtrend
    trend = np.where(close > supertrend_val, 1, -1)
    
    return supertrend_val, trend, atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 6h Supertrend (10, 3.0)
    st_6h, trend_6h, atr_6h = supertrend(high, low, close, period=10, multiplier=3.0)
    
    # 1d Supertrend for trend filter
    st_1d, trend_1d, atr_1d = supertrend(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=10, 
        multiplier=3.0
    )
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 6h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(trend_6h[i]) or np.isnan(trend_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # Entry conditions
        # Long: 6h uptrend AND 1d uptrend AND volume confirmation
        if trend_6h[i] == 1 and trend_1d_aligned[i] == 1 and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: 6h downtrend AND 1d downtrend AND volume confirmation
        elif trend_6h[i] == -1 and trend_1d_aligned[i] == -1 and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite 6h Supertrend signal
        elif position == 1 and trend_6h[i] == -1:
            position = 0
            signals[i] = 0.0
        elif position == -1 and trend_6h[i] == 1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals