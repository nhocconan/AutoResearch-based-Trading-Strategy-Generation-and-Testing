#!/usr/bin/env python3
"""
6h_Polarized_EMA_Fade_1dTrend
Hypothesis: In 6h timeframe, price often reverts to the 1-day EMA34 after extended moves.
Fade extreme deviations from 1d EMA34 (beyond 2 ATR) only when 12h trend agrees.
Use 12h EMA50 for trend filter and ATR for deviation measurement.
Designed for low turnover in ranging/trending markets with clear mean reversion edge.
"""

name = "6h_Polarized_EMA_Fade_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA34 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12h EMA50 Trend Direction ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_6h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === ATR(22) for Deviation Measurement ===
    tr1 = np.maximum(high[1:] - low[1:], 0)
    tr2 = np.maximum(np.abs(high[1:] - close[:-1]), 0)
    tr3 = np.maximum(np.abs(low[1:] - close[:-1]), 0)
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=22, adjust=False, min_periods=22).mean().values
    
    # === Signal Parameters ===
    position_size = 0.25
    deviation_multiplier = 2.0  # Fade when price deviates > 2*ATR from 1d EMA34
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for longest indicator)
    start_idx = 100  # covers EMA34 and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1d_6h[i]) or np.isnan(ema50_12h_6h[i]) or 
            np.isnan(atr[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate deviation from 1d EMA34
        deviation = close[i] - ema34_1d_6h[i]
        
        if position == 0:
            # Long: Price significantly below 1d EMA34 AND 12h trend is up
            if deviation < -deviation_multiplier * atr[i] and close[i] > ema50_12h_6h[i]:
                signals[i] = position_size
                position = 1
            # Short: Price significantly above 1d EMA34 AND 12h trend is down
            elif deviation > deviation_multiplier * atr[i] and close[i] < ema50_12h_6h[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price returns to touch or cross 1d EMA34
            if position == 1:
                if close[i] >= ema34_1d_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] <= ema34_1d_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals