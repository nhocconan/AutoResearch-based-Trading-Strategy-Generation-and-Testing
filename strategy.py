#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d EMA50 trend filter
# Donchian breakout captures momentum in both bull and bear markets
# Volume spike (>2.0 x 20-period EMA) confirms breakout validity and reduces false signals
# 1d EMA50 provides higher-timeframe trend alignment to avoid counter-trend trades
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag

name = "4h_Donchian20_Breakout_VolumeConfirm_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels - highest high and lowest low of last 20 periods
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Breakout conditions
    upper_break = close > high_roll  # Price breaks above upper channel
    lower_break = close < low_roll   # Price breaks below lower channel
    
    # 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_break[i]) or 
            np.isnan(lower_break[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Upper Donchian breakout with volume confirmation and uptrend
            if upper_break[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Lower Donchian breakout with volume confirmation and downtrend
            elif lower_break[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price retouches middle of channel OR trend changes to downtrend
            middle_channel = (high_roll[i] + low_roll[i]) / 2
            if close[i] < middle_channel or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price retouches middle of channel OR trend changes to uptrend
            middle_channel = (high_roll[i] + low_roll[i]) / 2
            if close[i] > middle_channel or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals