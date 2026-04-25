#!/usr/bin/env python3
"""
12h Camarilla H3L3 Breakout with 1d ATR Trend and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance on 12h timeframe. 
Breakouts above H3 or below L3 with volume confirmation and aligned 1d ATR-based trend 
capture institutional moves with lower frequency. The 1d ATR trend filter adapts to 
volatility regimes, working in both bull and bear markets. Designed for 12-37 trades/year 
to minimize fee drag while maintaining edge.
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
    
    # Get 1d data for ATR trend and Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ATR on 1d for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d trend: price > previous close + 0.5*ATR (uptrend), price < previous close - 0.5*ATR (downtrend)
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First period
    trend_up = close_1d > (prev_close_1d + 0.5 * atr_14_1d)
    trend_down = close_1d < (prev_close_1d - 0.5 * atr_14_1d)
    
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    
    # Get 1d data for Camarilla pivot calculation (H3, L3 levels)
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivots for each 1d bar: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla formulas:
    # H3 = close + (high - low) * 1.1/2
    # L3 = close - (high - low) * 1.1/2
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to LTF (12h) - no extra delay needed as pivots are based on completed 1d bar
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for ATR, volume MA, and to avoid NaN from shift
    start_idx = max(14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_val = atr_14_1d_aligned[i]
        trend_up_val = trend_up_aligned[i] > 0.5
        trend_down_val = trend_down_aligned[i] > 0.5
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 resistance AND volume spike AND uptrend
            long_entry = (curr_close > h3_level) and vol_spike and trend_up_val
            # Short: price breaks below L3 support AND volume spike AND downtrend
            short_entry = (curr_close < l3_level) and vol_spike and trend_down_val
            
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
            # Exit: price crosses below L3 support (broken support) OR downtrend emerges
            if (curr_close < l3_level) or trend_down_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 resistance (broken resistance) OR uptrend emerges
            if (curr_close > h3_level) or trend_up_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0