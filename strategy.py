#!/usr/bin/env python3
# 4h_12h_camarilla_pivot_volume_v1
# Hypothesis: Use 12h Camarilla pivot levels for entry, 12h trend filter for direction, and volume confirmation for institutional participation.
# Works in bull markets (trend continuation) and bear markets (mean reversion from pivot levels).
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) by requiring multi-timeframe alignment and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
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
    
    # Get 12h data for Camarilla pivots and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (using previous day's OHLC)
    # Camarilla levels: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low), L3 = Close - 1.125*(High-Low)
    # H2 = Close + 0.75*(High-Low), L2 = Close - 0.75*(High-Low)
    # H1 = Close + 0.5*(High-Low), L1 = Close - 0.5*(High-Low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    range_12h = high_12h - low_12h
    camarilla_h4 = close_12h + 1.5 * range_12h
    camarilla_l4 = close_12h - 1.5 * range_12h
    camarilla_h3 = close_12h + 1.125 * range_12h
    camarilla_l3 = close_12h - 1.125 * range_12h
    camarilla_h2 = close_12h + 0.75 * range_12h
    camarilla_l2 = close_12h - 0.75 * range_12h
    camarilla_h1 = close_12h + 0.5 * range_12h
    camarilla_l1 = close_12h - 0.5 * range_12h
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l1)
    
    # Trend filter: 12h EMA(20) for direction
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 2x average of last 24 periods (2 days in 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or \
           np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3 or below 12h EMA
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3 or above 12h EMA
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price touches Camarilla L3 with bullish trend and volume
            if (close[i] <= camarilla_l3_aligned[i] * 1.001 and  # Allow small tolerance
                close[i] > ema_12h_aligned[i] and  # Above 12h EMA (bullish trend)
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches Camarilla H3 with bearish trend and volume
            elif (close[i] >= camarilla_h3_aligned[i] * 0.999 and  # Allow small tolerance
                  close[i] < ema_12h_aligned[i] and  # Below 12h EMA (bearish trend)
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals