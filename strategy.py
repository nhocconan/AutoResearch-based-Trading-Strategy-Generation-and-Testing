#!/usr/bin/env python3
# 12h_camarilla_1w_trend_volume_v1
# Hypothesis: 12h Camarilla pivot levels with 1-week EMA50 trend filter and volume confirmation.
# Works in bull/bear: 1w EMA50 defines institutional trend; Camarilla levels from 1d provide precise
# entry/exit zones; volume confirms institutional participation. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d HTF data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low), L3 = Close - 1.125*(High-Low)
    # We'll use H3/L3 for entries and H4/L4 for stops
    rng = high_1d - low_1d
    h3 = close_1d + 1.125 * rng
    l3 = close_1d - 1.125 * rng
    h4 = close_1d + 1.5 * rng
    l4 = close_1d - 1.5 * rng
    
    # Align Camarilla levels to 12h timeframe (use previous completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR trend turns bearish OR stop at L4 hit
            if close[i] < l3_aligned[i] or close[i] < ema50_1w_aligned[i] or close[i] <= l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR trend turns bullish OR stop at H4 hit
            if close[i] > h3_aligned[i] or close[i] > ema50_1w_aligned[i] or close[i] >= h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price closes above H3 with bullish trend
                if close[i] > h3_aligned[i] and close[i] > ema50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price closes below L3 with bearish trend
                elif close[i] < l3_aligned[i] and close[i] < ema50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals