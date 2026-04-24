#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Camarilla pivot levels: Calculated from prior 1d OHLC (H4, L4 levels for breakout - tighter than H3/L3).
- Entry: Long when price breaks above prior 1d H4 AND 1d EMA50 bullish AND volume > 2.5 * volume MA(20).
         Short when price breaks below prior 1d L4 AND 1d EMA50 bearish AND volume > 2.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below 1d EMA50,
        exit short when price crosses above 1d EMA50.
- Signal size: 0.25 discrete to balance return and drawdown.
Uses 1d EMA50 trend filter (longer than EMA34) to reduce noise and improve trade quality in both bull and bear markets.
H4/L4 levels are tighter than H3/L3, requiring stronger breakouts for entry, reducing false signals.
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
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate prior 1d Camarilla levels (H4, L4 - tighter levels)
    # Using prior 1d candle to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations for H4/L4 levels (tighter than H3/L3)
    rang = high_1d - low_1d
    camarilla_h4 = close_1d + rang * 1.1 / 2  # H4 level
    camarilla_l4 = close_1d - rang * 1.1 / 2  # L4 level
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Need enough bars for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.5x threshold - stricter)
            vol_confirmed = curr_volume > 2.5 * vol_ma[i]
            
            # Long: Price breaks above prior 1d H4 AND 1d EMA50 bullish AND volume confirmed
            if curr_close > camarilla_h4_aligned[i] and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 1d L4 AND 1d EMA50 bearish AND volume confirmed
            elif curr_close < camarilla_l4_aligned[i] and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below 1d EMA50 (trend change)
            if curr_close < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above 1d EMA50 (trend change)
            if curr_close > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0