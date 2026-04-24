#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA50 trend alignment
- Camarilla levels calculated from prior 1d OHLC: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
- Long when price breaks above H3 with volume > 1.5*20-period volume MA and price > 1d EMA50
- Short when price breaks below L3 with volume > 1.5*20-period volume MA and price < 1d EMA50
- Exit: reverse signal or when price returns to 1d close (mean reversion)
- Discrete signal size: 0.25 to balance capture and fee drag
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla captures institutional levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate prior 1d Camarilla H3/L3 levels (using prior day's OHLC)
    # We need to shift the 1d data by 1 to avoid look-ahead
    if len(df_1d) >= 2:
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
        prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
        
        # Camarilla calculations
        camarilla_range = prev_high_aligned - prev_low_aligned
        h3 = prev_close_aligned + 1.1 * camarilla_range / 4
        l3 = prev_close_aligned - 1.1 * camarilla_range / 4
        pivot = prev_close_aligned  # for exit condition
    else:
        h3 = np.full(n, np.nan)
        l3 = np.full(n, np.nan)
        pivot = np.full(n, np.nan)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1d EMA50 and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(pivot[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND uptrend AND volume confirmation
            if close[i] > h3[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND downtrend AND volume confirmation
            elif close[i] < l3[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot (mean reversion) or reverse signal
            if close[i] <= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot (mean reversion) or reverse signal
            if close[i] >= pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0