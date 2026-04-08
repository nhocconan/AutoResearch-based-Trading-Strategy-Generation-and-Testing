#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_volume_v2
# Hypothesis: Use daily Camarilla pivot levels as support/resistance zones on 4h chart. Enter long when price touches/approaches L3 level with bullish momentum and volume confirmation, short at H3 level with bearish momentum. Uses 1d EMA200 as trend filter to avoid countertrend trades. Designed for fewer, high-quality trades in both bull and bear markets by focusing on institutional pivot levels with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v2"
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: range = high - low
    # H4 = close + range * 1.1/2
    # H3 = close + range * 1.1/4
    # H2 = close + range * 1.1/6
    # H1 = close + range * 1.1/12
    # L1 = close - range * 1.1/12
    # L2 = close - range * 1.1/6
    # L3 = close - range * 1.1/4
    # L4 = close - range * 1.1/2
    daily_range = high_1d - low_1d
    camarilla_h3 = close_1d + daily_range * 1.1 / 4
    camarilla_l3 = close_1d - daily_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Trend filter: 1d EMA200
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: current volume > 1.3x average of last 10 periods
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    # Momentum confirmation: price change over 3 periods
    price_change = (close - np.roll(close, 3)) / np.roll(close, 3)
    mom_confirm_long = price_change > 0
    mom_confirm_short = price_change < 0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(price_change[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 level or breaks below EMA200
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above H3 level or breaks above EMA200
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Tolerance for level touch: 0.1% of price level
            h3_tolerance = camarilla_h3_aligned[i] * 0.001
            l3_tolerance = camarilla_l3_aligned[i] * 0.001
            
            # Long entry: price near L3 level, above EMA200, with bullish momentum and volume
            if (abs(close[i] - camarilla_l3_aligned[i]) <= l3_tolerance and 
                close[i] > ema200_1d_aligned[i] and 
                mom_confirm_long[i] and vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price near H3 level, below EMA200, with bearish momentum and volume
            elif (abs(close[i] - camarilla_h3_aligned[i]) <= h3_tolerance and 
                  close[i] < ema200_1d_aligned[i] and 
                  mom_confirm_short[i] and vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals