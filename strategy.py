#!/usr/bin/env python3
"""
12H_TRIX_Trend_Volume_Confirmation_v1
Hypothesis: Use 1w TRIX for trend direction and 12h price action for entry.
Long when 1w TRIX > 0 and 12h price closes above 12h EMA20; 
Short when 1w TRIX < 0 and 12h price closes below 12h EMA20.
Volume confirmation: current volume > 1.5x 20-period average volume.
This combines long-term trend with medium-term momentum to capture sustained moves while avoiding whipsaws.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag in bear markets.
"""
name = "12H_TRIX_Trend_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for TRIX trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w TRIX (15-period triple EMA)
    close_1w = pd.Series(df_1w['close'])
    ema1 = close_1w.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    trix_signal = trix > 0  # Bullish when TRIX > 0
    trix_signal_aligned = align_htf_to_ltf(prices, df_1w, trix_signal.astype(float))
    
    # Get 12h data for EMA20
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20
    close_12h = pd.Series(df_12h['close'])
    ema20 = close_12h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_aligned = align_htf_to_ltf(prices, df_12h, ema20)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(trix_signal_aligned[i]) or np.isnan(ema20_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 20 bars between trades (10 days on 12h TF) to reduce frequency
            if bars_since_exit < 20:
                continue
                
            # Long: bullish TRIX and price above EMA20
            if (trix_signal_aligned[i] > 0.5 and 
                close[i] > ema20_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: bearish TRIX and price below EMA20
            elif (trix_signal_aligned[i] < 0.5 and 
                  close[i] < ema20_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: TRIX signal changes or price crosses EMA20 in opposite direction
            if position == 1 and (trix_signal_aligned[i] < 0.5 or close[i] < ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and (trix_signal_aligned[i] > 0.5 or close[i] > ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals