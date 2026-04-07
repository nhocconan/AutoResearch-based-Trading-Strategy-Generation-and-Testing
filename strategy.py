#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_trend_volume_v2
Hypothesis: Camarilla pivot levels from daily timeframe provide institutional support/resistance.
Trade breakouts of these levels with volume confirmation and daily trend filter.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
Targets 20-50 trades/year by requiring daily Camarilla level breakout + volume spike + daily trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v2"
timeframe = "4h"
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
    
    # Daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # H4 = C + (H-L) * 1.1/2
    # H3 = C + (H-L) * 1.1/4
    # H2 = C + (H-L) * 1.1/6
    # H1 = C + (H-L) * 1.1/12
    # L1 = C - (H-L) * 1.1/12
    # L2 = C - (H-L) * 1.1/6
    # L3 = C - (H-L) * 1.1/4
    # L4 = C - (H-L) * 1.1/2
    camarilla_h4 = typical_price + range_1d * 1.1 / 2.0
    camarilla_h3 = typical_price + range_1d * 1.1 / 4.0
    camarilla_h2 = typical_price + range_1d * 1.1 / 6.0
    camarilla_h1 = typical_price + range_1d * 1.1 / 12.0
    camarilla_l1 = typical_price - range_1d * 1.1 / 12.0
    camarilla_l2 = typical_price - range_1d * 1.1 / 6.0
    camarilla_l3 = typical_price - range_1d * 1.1 / 4.0
    camarilla_l4 = typical_price - range_1d * 1.1 / 2.0
    
    # Use H3 and L3 as primary breakout levels (most significant)
    breakout_high = camarilla_h3
    breakout_low = camarilla_l3
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align daily levels to 4h timeframe
    breakout_high_4h = align_htf_to_ltf(prices, df_1d, breakout_high)
    breakout_low_4h = align_htf_to_ltf(prices, df_1d, breakout_low)
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average on 4h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_4h[i]) or 
            np.isnan(breakout_high_4h[i]) or 
            np.isnan(breakout_low_4h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        vol_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below L3 OR trend turns down
            if close[i] < breakout_low_4h[i] or close[i] < ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above H3 OR trend turns up
            if close[i] > breakout_high_4h[i] or close[i] > ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above H3 + volume + uptrend
            if (close[i] > breakout_high_4h[i] and 
                vol_confirm and 
                close[i] > ema50_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below L3 + volume + downtrend
            elif (close[i] < breakout_low_4h[i] and 
                  vol_confirm and 
                  close[i] < ema50_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals