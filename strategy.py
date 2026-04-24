#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 Breakout with 12h EMA50 Trend Filter and Volume Spike.
- Primary timeframe: 4h for execution, HTF: 12h for EMA50 trend filter.
- Entry: Price breaks above Camarilla H3 (long) or below L3 (short) on 4h close, with volume > 1.8x 20-period volume MA.
- Direction filter: only long when 4h close > 12h EMA50, only short when 4h close < 12h EMA50.
- Camarilla levels from 1d provide strong intraday support/resistance; EMA50 filters for trend alignment.
- Volume confirmation reduces false breakouts.
- Exit: Price returns to Camarilla Pivot Point (PP) or trend filter reversal.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d Camarilla levels (based on previous 1d OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift OHLC by 1 to use previous day's data (avoid look-ahead)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    close_1d_shifted = np.roll(close_1d, 1)
    # First bar: use same day's data (no prior day available)
    high_1d_shifted[0] = high_1d[0]
    low_1d_shifted[0] = low_1d[0]
    close_1d_shifted[0] = close_1d[0]
    
    # Camarilla calculations: based on previous day's range
    rng = high_1d_shifted - low_1d_shifted
    camarilla_pp = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3
    camarilla_h3 = camarilla_pp + 1.1 * (high_1d_shifted - low_1d_shifted) / 2
    camarilla_l3 = camarilla_pp - 1.1 * (high_1d_shifted - low_1d_shifted) / 2
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1  # Need 12h EMA50, volume MA(20), plus 1 for Camarilla shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla H3 with volume spike AND uptrend (close > 12h EMA50)
            if (close[i] > camarilla_h3_aligned[i] and volume_spike[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla L3 with volume spike AND downtrend (close < 12h EMA50)
            elif (close[i] < camarilla_l3_aligned[i] and volume_spike[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to Camarilla Pivot Point or trend reversal
            if (close[i] < camarilla_pp_aligned[i] or close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to Camarilla Pivot Point or trend reversal
            if (close[i] > camarilla_pp_aligned[i] or close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0