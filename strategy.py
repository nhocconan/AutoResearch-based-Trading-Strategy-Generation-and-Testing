#!/usr/bin/env python3
"""
12h_1w_1d_camarilla_breakout_volume_v1
Hypothesis: Use weekly trend (EMA21) to filter direction, daily Camarilla pivot levels for entry/exit, and volume confirmation on 12h timeframe.
Camarilla levels provide high-probability support/resistance. Breakouts in direction of weekly trend with volume capture institutional moves.
Works in bull/bear via trend filter. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Camarilla pivot levels: based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # H4 = Close + 1.5 * (High - Low) * 1.1/2
    # L4 = Close - 1.5 * (High - Low) * 1.1/2
    # H3 = Close + 1.25 * (High - Low) * 1.1/2
    # L3 = Close - 1.25 * (High - Low) * 1.1/2
    # H2 = Close + 1.083 * (High - Low) * 1.1/2
    # L2 = Close - 1.083 * (High - Low) * 1.1/2
    # H1 = Close + 1.0/2 * (High - Low) * 1.1/2
    # L1 = Close - 1.0/2 * (High - Low) * 1.1/2
    
    # We'll use H3 and L3 as primary resistance/support (widely watched)
    camarilla_h3 = close_1d + 1.25 * (high_1d - low_1d) * 1.1 / 2
    camarilla_l3 = close_1d - 1.25 * (high_1d - low_1d) * 1.1 / 2
    
    # Forward fill to get the most recent Camarilla level
    camarilla_h3_series = pd.Series(camarilla_h3)
    camarilla_l3_series = pd.Series(camarilla_l3)
    camarilla_h3_ffilled = camarilla_h3_series.ffill().values
    camarilla_l3_ffilled = camarilla_l3_series.ffill().values
    
    # Align to 12h with 1-bar delay for daily close (Camarilla based on previous day's close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_ffilled)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_ffilled)
    
    # Volume confirmation: volume > 1.3x average of last 8 periods (8*12h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=8, min_periods=8).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to Camarilla L3 support or trend changes
            if close[i] <= camarilla_l3_aligned[i] or ema_1w_aligned[i] < ema_1w_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla H3 resistance or trend changes
            if close[i] >= camarilla_h3_aligned[i] or ema_1w_aligned[i] > ema_1w_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above Camarilla H3 resistance with volume and weekly uptrend
            if (not np.isnan(camarilla_h3_aligned[i]) and close[i] > camarilla_h3_aligned[i] and 
                ema_1w_aligned[i] > ema_1w_aligned[max(0, i-1)] and  # Uptrend confirmation
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 support with volume and weekly downtrend
            elif (not np.isnan(camarilla_l3_aligned[i]) and close[i] < camarilla_l3_aligned[i] and 
                  ema_1w_aligned[i] < ema_1w_aligned[max(0, i-1)] and  # Downtrend confirmation
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals