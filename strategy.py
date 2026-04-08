#!/usr/bin/env python3
"""
1d_1w_ema_trend_volume_v1
Hypothesis: Use weekly EMA for long-term bias and daily close for entry with volume confirmation.
Long when daily close crosses above weekly EMA with volume and weekly bullish alignment.
Short when daily close crosses below weekly EMA with volume and weekly bearish alignment.
Designed to capture major trends while avoiding whipsaws in ranging markets.
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily EMA (21-period)
    close_1d = df_1d['close'].values
    ema_21 = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Weekly EMA (13-period) for trend bias
    close_1w = df_1w['close'].values
    ema_13 = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1w, ema_13)
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_21_aligned[i]) or np.isnan(ema_13_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below daily EMA21 or weekly EMA turns bearish
            if close[i] < ema_21_aligned[i] or ema_13_aligned[i] < ema_13_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above daily EMA21 or weekly EMA turns bullish
            if close[i] > ema_21_aligned[i] or ema_13_aligned[i] > ema_13_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price crosses above daily EMA21 with volume and weekly bullish
            if close[i] > ema_21_aligned[i] and vol_confirm[i] and ema_13_aligned[i] > ema_13_aligned[i-1]:
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below daily EMA21 with volume and weekly bearish
            elif close[i] < ema_21_aligned[i] and vol_confirm[i] and ema_13_aligned[i] < ema_13_aligned[i-1]:
                position = -1
                signals[i] = -0.25
    
    return signals