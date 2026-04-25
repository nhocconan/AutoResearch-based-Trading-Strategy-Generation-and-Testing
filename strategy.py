#!/usr/bin/env python3
"""
6h Weekly Camarilla H3/L3 Breakout + 1d Volume Spike + 12h EMA50 Trend
Hypothesis: Weekly Camarilla H3 and L3 levels act as strong support/resistance.
Breakouts above H3 or below L3 with volume spike and 12h EMA50 trend alignment
capture institutional moves. Works in bull markets (continuation) and bear 
markets (failed breakouts, reversals to H3/L3). Weekly HTF loaded ONCE.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
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
    open_ = prices['open'].values
    
    # ATR for stop (optional, using signal=0 for exit)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Weekly data for Camarilla pivot points (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    # Weekly Camarilla: based on previous week's OHLC
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    weekly_range = prev_week_high - prev_week_low
    # Camarilla levels (H3/L3 are most important)
    weekly_h3 = prev_week_close + weekly_range * 1.1 / 4
    weekly_l3 = prev_week_close - weekly_range * 1.1 / 4
    weekly_h4 = prev_week_close + weekly_range * 1.1 / 2
    weekly_l4 = prev_week_close - weekly_range * 1.1 / 2
    weekly_h3_aligned = align_htf_to_ltf(prices, df_1w, weekly_h3)
    weekly_l3_aligned = align_htf_to_ltf(prices, df_1w, weekly_l3)
    weekly_h4_aligned = align_htf_to_ltf(prices, df_1w, weekly_h4)
    weekly_l4_aligned = align_htf_to_ltf(prices, df_1w, weekly_l4)
    
    # 12h EMA50 trend filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly/daily data and volume MA
    start_idx = max(51, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_h3_aligned[i]) or np.isnan(weekly_l3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        breakout_long = curr_close > weekly_h3_aligned[i]
        breakout_short = curr_close < weekly_l3_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: weekly Camarilla breakout + volume spike + 12h EMA50 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_12h_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_12h_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on weekly L3 retrace or trend change
            if curr_close < weekly_l3_aligned[i] or curr_close < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on weekly H3 retrace or trend change
            if curr_close > weekly_h3_aligned[i] or curr_close > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyCamarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0