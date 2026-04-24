#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d trend filter (price > 1d EMA50 for long, < for short) and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian channel: 20-period high/low from 6h data for breakout detection.
- Entry: Long when price breaks above prior 20-bar 6h high AND 1d EMA50 bullish AND volume > 1.5 * volume MA(20).
         Short when price breaks below prior 20-bar 6h low AND 1d EMA50 bearish AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below prior 20-bar 6h low,
        exit short when price crosses above prior 20-bar 6h high.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to capture medium-term trends with noise filtering via volume and trend confirmation.
Works in bull markets via breakout longs and in bear markets via breakdown shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channel (20-period) from 6h data
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA(20) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need enough bars for EMA50, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above prior 20-bar 6h high AND 1d EMA50 bullish AND volume confirmed
            if curr_high > high_ma[i-1] and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 20-bar 6h low AND 1d EMA50 bearish AND volume confirmed
            elif curr_low < low_ma[i-1] and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior 20-bar 6h low (reversion to mean)
            if curr_close < low_ma[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior 20-bar 6h high (reversion to mean)
            if curr_close > high_ma[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0