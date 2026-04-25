#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance on 12h timeframe. 
Breakouts above H3 or below L3 with volume confirmation and 1d EMA34 trend filter capture 
institutional participation. Works in both bull/bear markets by trend-filtering breakouts.
Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day (H3/L3 levels)
    # Shift to get previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First bar: use current values (will be filtered out by min_periods anyway)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla pivot calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla H3 and L3 levels (stronger than R1/S1)
    H3 = pivot + (range_hl * 1.1 / 4)  # H3 level
    L3 = pivot - (range_hl * 1.1 / 4)  # L3 level
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(H3[i]) or np.isnan(L3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        h3_level = H3[i]
        l3_level = L3[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout signals with trend filter
        if position == 0:
            # Long: price breaks above H3 AND above 1d EMA34 (uptrend filter)
            long_condition = (curr_close > h3_level) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below L3 AND below 1d EMA34 (downtrend filter)
            short_condition = (curr_close < l3_level) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot or trend breaks
            if curr_close <= pivot[i] or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot or trend breaks
            if curr_close >= pivot[i] or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3_L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0