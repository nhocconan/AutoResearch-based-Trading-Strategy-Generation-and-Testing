#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels represent strong intraday support/resistance. 
Breakouts beyond these levels with 12h EMA50 trend alignment and volume confirmation 
capture institutional participation. Discrete sizing (0.30) and ATR trailing stop (2.5x) 
limit fee drag. Designed for both bull (breakouts with trend) and bear (failed breaks, reversals) markets.
Target: 20-50 trades/year per symbol.
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
    
    # ATR for trailing stop
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 12h EMA50 trend filter (MTF) - loaded ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h OHLC for Camarilla pivot levels (using previous 12h bar's data)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_range = high_12h - low_12h
    h3 = close_12h + (1.1 * camarilla_range / 4)
    l3 = close_12h - (1.1 * camarilla_range / 4)
    
    # Align Camarilla levels to 4h timeframe (12h -> 4h)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(20, 50) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Camarilla breakout conditions (use previous bar's levels to avoid look-ahead)
        breakout_long = curr_close > h3_aligned[i-1]
        breakout_short = curr_close < l3_aligned[i-1]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + volume spike + 12h EMA50 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_12h_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_12h_aligned[i])
            
            if long_entry:
                signals[i] = 0.30
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.30
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
                signals[i] = 0.30
        elif position == -1:
            # Short position management: ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.5 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0