#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_trend_volume_v2
Hypothesis: Weekly Camarilla pivot levels act as major support/resistance.
Price breaking above/below these levels with volume confirmation and weekly trend alignment
captures institutional breakouts in both bull and bear markets. Weekly trend filter reduces
whipsaws. Targets 7-25 trades/year by requiring confluence of weekly Camarilla break,
volume spike, and weekly trend filter. Uses daily timeframe for execution.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly OHLC for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week's OHLC
    prev_close = df_1w['close'].values
    prev_high = df_1w['high'].values
    prev_low = df_1w['low'].values
    
    # Calculate range
    weekly_range = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h4 = prev_close + 1.5 * weekly_range
    camarilla_l4 = prev_close - 1.5 * weekly_range
    camarilla_h3 = prev_close + 1.0 * weekly_range
    camarilla_l3 = prev_close - 1.0 * weekly_range
    camarilla_h2 = prev_close + 0.5 * weekly_range
    camarilla_l2 = prev_close - 0.5 * weekly_range
    camarilla_h1 = prev_close + 0.25 * weekly_range
    camarilla_l1 = prev_close - 0.25 * weekly_range
    
    # Align to daily timeframe (shift by 1 week for completed bars only)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l1)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(prev_close).ewm(span=50, adjust=False).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_1d[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Camarilla L3 OR trend turns down
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla H3 OR trend turns up
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Camarilla H4 + volume + uptrend
            if (close[i] > camarilla_h4_aligned[i] and 
                vol_confirm and 
                close[i] > ema50_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Camarilla L4 + volume + downtrend
            elif (close[i] < camarilla_l4_aligned[i] and 
                  vol_confirm and 
                  close[i] < ema50_1d[i]):
                position = -1
                signals[i] = -0.25
    
    return signals