#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout with 1d EMA50 Trend and Volume Spike
Hypothesis: Camarilla H3/L3 levels (stronger daily support/resistance) breakouts with volume confirmation and 1d EMA50 trend alignment capture significant swing moves. ATR trailing stop manages risk. Designed for fewer trades (~30-50/year) to minimize fee drag and work in both bull/bear markets via trend filter.
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
    
    # ATR for volatility and trailing stop
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average (slightly looser for more signals but still controlled)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # 1d EMA50 trend filter (MTF) - loaded ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Camarilla pivot levels from daily OHLC (MTF) - H3/L3 levels
    camarilla_h3 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 4
    camarilla_l3 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 4
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(20, 50) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions: price breaks Camarilla H3 or L3 levels
        breakout_long = curr_close > h3_aligned[i]
        breakout_short = curr_close < l3_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: H3/L3 breakout + volume spike + 1d EMA50 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_1d_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management: ATR trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            exit_level = highest_since_entry - (2.5 * atr_14[i])
            
            if curr_close < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management: ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.5 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0