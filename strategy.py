#!/usr/bin/env python3
"""
12h_1d_1w_camarilla_breakout_volume_v1
Hypothesis: Use 12h price action with 1d Camarilla pivot levels and 1w trend bias.
Long when 12h price breaks above 1d H3 with 1w bullish trend and volume confirmation.
Short when 12h price breaks below 1d L3 with 1w bearish trend and volume confirmation.
Designed to capture institutional breakouts at key 1d Camarilla levels with trend alignment.
Target: 12-30 trades/year per symbol (48-120 total over 4 years) by requiring strong breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
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
    
    # Get 1w data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, L3, L4
    # Based on previous day's range
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + range_1d * 1.1 / 4
    camarilla_l3 = close_1d - range_1d * 1.1 / 4
    camarilla_h4 = close_1d + range_1d * 1.1 / 2
    camarilla_l4 = close_1d - range_1d * 1.1 / 2
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1w trend bias: close > EMA(20) for bullish, close < EMA(20) for bearish
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_bullish = close_1w > ema_20
    trend_bearish = close_1w < ema_20
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish.astype(float))
    
    # Volume confirmation: volume > 1.5x average of last 10 periods
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(l4_aligned[i]) or np.isnan(trend_bullish_aligned[i]) or
            np.isnan(trend_bearish_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 1d L3 or 1w trend turns bearish
            if close[i] < l3_aligned[i] or trend_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above 1d H3 or 1w trend turns bullish
            if close[i] > h3_aligned[i] or trend_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 1d H3 with 1w bullish trend and volume
            if close[i] > h3_aligned[i] and trend_bullish_aligned[i] > 0.5 and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 1d L3 with 1w bearish trend and volume
            elif close[i] < l3_aligned[i] and trend_bearish_aligned[i] > 0.5 and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals