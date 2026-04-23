#!/usr/bin/env python3
"""
12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
- Camarilla levels (R1, S1) from prior 1d: strong intraday support/resistance
- Long: price breaks above R1 + volume > 1.8x 24-period avg + price > 1d EMA50
- Short: price breaks below S1 + volume > 1.8x 24-period avg + price < 1d EMA50
- Exit: price re-enters Camarilla H1-L1 range (mean reversion) OR EMA50 trend flip
- Uses Camarilla for structure, volume for conviction, 1d EMA50 for HTF filter
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
"""

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
    
    # Volume confirmation: > 1.8x 24-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla levels (based on prior 1d OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values  # same as close_1d, but conceptually prior day
    
    # Camarilla formula: range = high - low
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # H1 = close + (high - low) * 1.1/6
    # L1 = close - (high - low) * 1.1/6
    rng = high_1d - low_1d
    camarilla_r1 = close_1d_prev + rng * (1.1 / 12)
    camarilla_s1 = close_1d_prev - rng * (1.1 / 12)
    camarilla_h1 = close_1d_prev + rng * (1.1 / 6)
    camarilla_l1 = close_1d_prev - rng * (1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)  # Need 50 for EMA50, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(h1_aligned[i]) or
            np.isnan(l1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + price > 1d EMA50
            if (close[i] > r1_aligned[i] and 
                volume_confirm and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + price < 1d EMA50
            elif (close[i] < s1_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below H1 (mean reversion) OR price < 1d EMA50 (trend flip)
            if close[i] < h1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above L1 (mean reversion) OR price > 1d EMA50 (trend flip)
            if close[i] > l1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0