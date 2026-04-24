#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h to capture medium-term swings while minimizing fee drag.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 12h volume > 2.0 * 20-period 12h volume MA to capture institutional interest.
- Camarilla: Calculate H3 (resistance) and L3 (support) levels from prior 1d OHLC.
- Entry: Long when close crosses above H3 AND 1d EMA34 bullish AND volume spike.
         Short when close crosses below L3 AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Camarilla level (L3 for long, H3 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This strategy exploits institutional order flow around key Camarilla pivot levels,
filtered by daily trend and volume confirmation. Works in both bull and bear markets
by only taking trades in the direction of the 1d trend, reducing whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from prior 1d OHLC (H3, L3)
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close_val = df_1d['close'].values
    camarilla_range = df_1d_high - df_1d_low
    H3 = df_1d_close_val + 1.1 * camarilla_range / 4.0
    L3 = df_1d_close_val - 1.1 * camarilla_range / 4.0
    
    # Calculate 20-period 12h volume MA for volume confirmation
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period 12h volume MA
    volume_spike = volume > (2.0 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: close crosses above H3 AND 1d EMA34 bullish (close > EMA)
                if curr_close > H3_aligned[i] and close[i-1] <= H3_aligned[i-1] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: close crosses below L3 AND 1d EMA34 bearish (close < EMA)
                elif curr_close < L3_aligned[i] and close[i-1] >= L3_aligned[i-1] and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: close crosses below L3 OR loss of volume confirmation
            if curr_close < L3_aligned[i] and close[i-1] >= L3_aligned[i-1] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close crosses above H3 OR loss of volume confirmation
            if curr_close > H3_aligned[i] and close[i-1] <= H3_aligned[i-1] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0