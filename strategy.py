#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: Camarilla pivot levels derived from daily price action act as key support/resistance.
Price breaking above/below these levels with volume confirmation and daily trend alignment
captures institutional breakouts. Works in bull markets (breakouts continue) and bear
markets (breakdowns continue) by trading with the daily trend. Targets 12-37 trades/year
by requiring confluence of Camarilla level break, volume spike, and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timezone = "12h"
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
    
    # Daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Formulas: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    #           H3 = Close + 1.0*(High-Low), L3 = Close - 1.0*(High-Low)
    #           H2 = Close + 0.5*(High-Low), L2 = Close - 0.5*(High-Low)
    #           H1 = Close + 0.25*(High-Low), L1 = Close - 0.25*(High-Low)
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Calculate range
    daily_range = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h4 = prev_close + 1.5 * daily_range
    camarilla_l4 = prev_close - 1.5 * daily_range
    camarilla_h3 = prev_close + 1.0 * daily_range
    camarilla_l3 = prev_close - 1.0 * daily_range
    camarilla_h2 = prev_close + 0.5 * daily_range
    camarilla_l2 = prev_close - 0.5 * daily_range
    camarilla_h1 = prev_close + 0.25 * daily_range
    camarilla_l1 = prev_close - 0.25 * daily_range
    
    # Align to 12h timeframe (shift by 1 day for completed bars only)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(prev_close).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_12h[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Camarilla L3 OR trend turns down
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla H3 OR trend turns up
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Camarilla H4 + volume + uptrend
            if (close[i] > camarilla_h4_aligned[i] and 
                vol_confirm and 
                close[i] > ema50_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Camarilla L4 + volume + downtrend
            elif (close[i] < camarilla_l4_aligned[i] and 
                  vol_confirm and 
                  close[i] < ema50_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals