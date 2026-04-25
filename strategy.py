#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance. 
Breakouts above H3 or below L3 with volume confirmation and 1d EMA34 trend filter capture 
momentum moves. Works in both bull and bear markets by trading breakouts in trend direction.
Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-37 trades/year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First bar has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    rang = prev_high - prev_low
    H3 = prev_close + 1.1 * rang / 4
    L3 = prev_close - 1.1 * rang / 4
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Load 1d EMA34 trend ONCE before loop (MTF Rule 1)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        # Trend filter: price above/below 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND uptrend
            long_entry = (curr_close > H3[i]) and vol_spike and uptrend
            # Short: price breaks below L3 AND volume spike AND downtrend
            short_entry = (curr_close < L3[i]) and vol_spike and downtrend
            
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
            # Exit: price falls below L3 (breakdown) OR loss of uptrend
            if (curr_close < L3[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 (breakout) OR loss of downtrend
            if (curr_close > H3[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0