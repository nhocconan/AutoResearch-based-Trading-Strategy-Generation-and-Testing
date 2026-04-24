#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses 4h timeframe (primary) and 1d HTF for EMA34 trend alignment
- Camarilla pivot levels calculated from previous 1d OHLC: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
- Breakout logic: long when close crosses above R3 from below, short when close crosses below S3 from above
- Trend filter: only long when price > 1d EMA34, only short when price < 1d EMA34
- Volume confirmation: current volume > 2.0 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when price touches Camarilla H3/L3 levels (mean reversion)
- Discrete signal size: 0.25 to balance profit potential and risk management
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture institutional levels
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
    open_time = prices['open_time'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d OHLC
    # Need to shift by 1 to use previous day's data (no look-ahead)
    df_1d_shifted = df_1d.copy()
    df_1d_shifted['high'] = df_1d_shifted['high'].shift(1)
    df_1d_shifted['low'] = df_1d_shifted['low'].shift(1)
    df_1d_shifted['close'] = df_1d_shifted['close'].shift(1)
    
    # Calculate Camarilla levels: R4, R3, H3, L3, S3, S4
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # H3 = close + 1.1*(high-low)*1.1/6
    # L3 = close - 1.1*(high-low)*1.1/6
    high_1d = df_1d_shifted['high'].values
    low_1d = df_1d_shifted['low'].values
    close_1d = df_1d_shifted['close'].values
    
    # Calculate range and Camarilla levels
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rng * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * rng * 1.1 / 4
    camarilla_h3 = close_1d + 1.1 * rng * 1.1 / 6
    camarilla_l3 = close_1d - 1.1 * rng * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 1d EMA34 and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close crosses above R3 from below AND uptrend AND volume confirmation
            if close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: close crosses below S3 from above AND downtrend AND volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches H3 level (mean reversion) or reverse signal
            if close[i] >= camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches L3 level (mean reversion) or reverse signal
            if close[i] <= camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0