#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance on 4h.
Breakouts above H3 or below L3 with volume spike and 1d EMA34 trend alignment capture
institutional moves. Works in bull/bear via 1d EMA34 trend filter (only trade in trend direction).
Designed for 75-200 trades over 4 years on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels (H3, L3, H4, L4)"""
    # Camarilla formulas based on previous day's range
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    high_val = high[-1]  # previous period's high
    low_val = low[-1]    # previous period's low
    close_val = close[-1] # previous period's close
    
    range_val = high_val - low_val
    h3 = close_val + 1.25 * range_val
    l3 = close_val - 1.25 * range_val
    h4 = close_val + 1.5 * range_val
    l4 = close_val - 1.5 * range_val
    
    return h3, l3, h4, l4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA34 trend filter and Camarilla levels (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Pre-calculate Camarilla levels for each 1d bar
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):  # start from 1 to have previous bar
        h3, l3, _, _ = calculate_camarilla(
            df_1d['high'].values[i-1],
            df_1d['low'].values[i-1],
            df_1d['close'].values[i-1]
        )
        camarilla_h3[i] = h3
        camarilla_l3[i] = l3
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA (20) + safety buffer
    start_idx = 25
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + volume spike + 1d EMA34 trend alignment
            long_entry = (curr_close > camarilla_h3_aligned[i] and 
                         vol_ma[i] > 0 and volume_spike[i] and 
                         (curr_close > ema_34_1d_aligned[i]))
            short_entry = (curr_close < camarilla_l3_aligned[i] and 
                          vol_ma[i] > 0 and volume_spike[i] and 
                          (curr_close < ema_34_1d_aligned[i]))
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below L3 or trend turns bearish
            if curr_close < camarilla_l3_aligned[i] or (curr_close < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above H3 or trend turns bullish
            if curr_close > camarilla_h3_aligned[i] or (curr_close > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0