#!/usr/bin/env python3
"""
12h_1d_vortex_breakout_volume_v1
Strategy: 12h Vortex indicator breakout with volume confirmation and 1d trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses Vortex indicator (VI+ and VI-) to detect trend direction and breakouts. 
Enters long when VI+ crosses above VI- with volume confirmation in uptrend (price > 1d EMA50).
Enters short when VI- crosses above VI+ with volume confirmation in downtrend (price < 1d EMA50).
Vortex is effective in catching trend changes and works in both bull and bear markets by 
identifying directional movement. Volume confirmation reduces false signals. 
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_vortex_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_vortex(high, low, close, period=14):
    """Calculate Vortex Indicator (VI+ and VI-)"""
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # VM+ and VM-
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    
    # Sum over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    return vi_plus, vi_minus

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Vortex indicator
    vi_plus, vi_minus = calculate_vortex(high, low, close, period=14)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price_close > ema_50_1d_aligned[i]
        downtrend = price_close < ema_50_1d_aligned[i]
        
        # Vortex crossovers
        vi_plus_cross_above = vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]
        vi_minus_cross_above = vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: VI+ crosses above VI- with volume in uptrend
        long_signal = vi_plus_cross_above and vol_confirmed and uptrend
        
        # Short: VI- crosses above VI+ with volume in downtrend
        short_signal = vi_minus_cross_above and vol_confirmed and downtrend
        
        # Exit when Vortex crosses in opposite direction
        exit_long = position == 1 and vi_minus_cross_above
        exit_short = position == -1 and vi_plus_cross_above
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals