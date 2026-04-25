#!/usr/bin/env python3
"""
6h Camarilla R4S4 Breakout with 1d EMA50 Trend and Volume Spike
Hypothesis: Camarilla R4/S4 levels represent strong breakout levels from the prior day's range.
Breakouts above R4 or below S4 with 1d EMA50 trend alignment and volume confirmation capture
institutional momentum moves. Works in both bull and bear markets by trading breakouts in the
direction of the higher timeframe trend. 6h timeframe targets 12-37 trades/year (50-150 total).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50)  # volume MA, 1d EMA alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current 6h bar using prior bar's OHLC
        if i == 0:
            continue
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        rang = prev_high - prev_low
        
        # Camarilla R4 and S4 levels (breakout levels)
        r4 = prev_close + rang * 1.1 / 2
        s4 = prev_close - rang * 1.1 / 2
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = ema_50_aligned[i] is not None and curr_close > ema_50_aligned[i]
        downtrend = ema_50_aligned[i] is not None and curr_close < ema_50_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: break above R4 AND uptrend AND volume spike
            long_entry = (curr_close > r4) and uptrend and vol_spike
            # Short: break below S4 AND downtrend AND volume spike
            short_entry = (curr_close < s4) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below S4 (reversal) OR loss of uptrend
            if (curr_close < s4) or (curr_close < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above R4 (reversal) OR loss of downtrend
            if (curr_close > r4) or (curr_close > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1dEMA50_Trend_VolumeSp"
timeframe = "6h"
leverage = 1.0