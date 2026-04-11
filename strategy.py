#!/usr/bin/env python3
"""
12h_1d_vortex_breakout_volume_v2
Strategy: 12h Vortex indicator breakout with volume confirmation and 1d trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Vortex indicator (VI+ and VI-) identifies trend direction. Breakouts above/below recent high/low with volume confirmation and aligned 1d trend filter capture strong moves. Designed to work in both bull and bear markets by filtering trades with higher timeframe trend. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_vortex_breakout_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Vortex indicator (14-period) on 12h data
    # VI+ = sum(|current high - prior low|) / sum(true range)
    # VI- = sum(|current low - prior high|) / sum(true range)
    tr1 = np.abs(high - np.roll(low, 1))
    tr2 = np.abs(low - np.roll(high, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # Sum over 14 periods
    n_periods = 14
    sum_vm_plus = pd.Series(vm_plus).rolling(window=n_periods, min_periods=n_periods).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=n_periods, min_periods=n_periods).sum().values
    sum_tr = pd.Series(tr).rolling(window=n_periods, min_periods=n_periods).sum().values
    
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # Recent high/low for breakout (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is invalid
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA34
        uptrend_1d = price_close > ema_34_1d_aligned[i]
        downtrend_1d = price_close < ema_34_1d_aligned[i]
        
        # Vortex trend: VI+ > VI- indicates uptrend, VI- > VI+ indicates downtrend
        vortex_up = vi_plus[i] > vi_minus[i]
        vortex_down = vi_minus[i] > vi_plus[i]
        
        # Breakout conditions
        breakout_up = price_close > highest_high[i]
        breakout_down = price_close < lowest_low[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: bullish vortex + upward breakout with volume in uptrend
        long_signal = vortex_up and breakout_up and vol_confirmed and uptrend_1d
        
        # Short: bearish vortex + downward breakout with volume in downtrend
        short_signal = vortex_down and breakout_down and vol_confirmed and downtrend_1d
        
        # Exit when vortex reverses or price returns to recent opposite extreme
        exit_long = position == 1 and (vi_plus[i] < vi_minus[i] or price_close < lowest_low[i])
        exit_short = position == -1 and (vi_minus[i] < vi_plus[i] or price_close > highest_high[i])
        
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