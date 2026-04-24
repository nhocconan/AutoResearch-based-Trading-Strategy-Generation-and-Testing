#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Camarilla pivot: Calculated from prior 4h OHLC (using prior bar's high/low/close).
- Entry: Long when price breaks above H3 AND 12h EMA50 bullish AND volume > 2.0 * volume MA(20).
         Short when price breaks below L3 AND 12h EMA50 bearish AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below L3,
        exit short when price crosses above H3.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to work in both bull and bear markets via trend filter and mean-reversion exits.
Proven pattern from DB: Camarilla breakouts with volume and trend filters show strong test performance.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for Camarilla pivot calculation (prior bar OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla calculations (using prior bar)
    pivot = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    
    # H3 and L3 levels (Camarilla)
    h3 = pivot + (range_4h * 1.1 / 4)
    l3 = pivot - (range_4h * 1.1 / 4)
    
    # Align HTF indicators to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 2, 20)  # Need enough bars for EMA50, Camarilla, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above H3 AND 12h EMA50 bullish AND volume confirmed
            if curr_close > h3_aligned[i] and curr_close > ema_12h_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 AND 12h EMA50 bearish AND volume confirmed
            elif curr_close < l3_aligned[i] and curr_close < ema_12h_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below L3 (reversion to mean)
            if curr_close < l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above H3 (reversion to mean)
            if curr_close > h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0